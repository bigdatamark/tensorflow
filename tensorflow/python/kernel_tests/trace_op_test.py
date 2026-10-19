# [MODULE DOC]
# Unit tests for the matrix trace op (sum of the diagonal over the last two axes), comparing
# the TensorFlow result against the NumPy reference across dtypes and shapes. Runs via the
# standard test main entry point when executed as a script. No import-time side effects.
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

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class TraceTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the trace op. Seeds NumPy's random generator in setUp and checks the op's
  # output against NumPy's trace over the last two axes for several dtypes and shapes.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: assigns the result of the seeding call (nothing) to an unused local variable,
    # and does not call the parent setUp.
    # Fixture run before each test: seeds the global NumPy random generator with zero so the
    # random inputs are reproducible. Side effect: mutates global NumPy random state.
    x = np.random.seed(0)

  def compare(self, x):
    # [FUNCTION DOC]
    # HELPER: shared comparison routine for the trace tests.
    # Takes a NumPy array x (at least two dimensions). Computes the reference trace over the
    # last two axes with NumPy, evaluates the TensorFlow trace op on x in a cached session, and
    # asserts the two results are close. Returns nothing.
    np_ans = np.trace(x, axis1=-2, axis2=-1)
    with self.cached_session():
      tf_ans = math_ops.trace(x).eval()
    self.assertAllClose(tf_ans, np_ans)

  @test_util.run_deprecated_v1
  def testTrace(self):
    # [FUNCTION DOC]
    # Graph-mode (deprecated v1) test. For int32, float32 and float64 and for shapes ranging
    # from 2 by 2 and non-square matrices to batched 3-D and 4-D arrays, generates random data
    # of that dtype and shape and compares the trace op with NumPy through the compare helper.
    for dtype in [np.int32, np.float32, np.float64]:
      for shape in [[2, 2], [2, 3], [3, 2], [2, 3, 2], [2, 2, 2, 3]]:
        x = np.random.rand(np.prod(shape)).astype(dtype).reshape(shape)
        self.compare(x)


if __name__ == "__main__":
  test.main()
