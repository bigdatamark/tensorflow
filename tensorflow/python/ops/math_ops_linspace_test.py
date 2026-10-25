# [MODULE DOC]
# Purpose: unit tests for the n-dimensional linspace function in the math ops module. It checks that
# start/stop broadcasting matches numpy along every axis and that static shape information is kept
# when the function is traced with constant count and axis.
# Exposes one test class; running the file as a script starts the standard test runner.
# Import-time side effects: none beyond imports.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.math_ops.linspace."""

from absl.testing import parameterized
import numpy as np
from packaging.version import Version  # pylint: disable=g-importing-member

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class LinspaceTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the n-dimensional linspace op. Every test method runs in both graph and
  # eager modes (class decorator). Combines the framework test base class with parameterized
  # test support. No shared fixtures.

  # pylint: disable=g-complex-comprehension
  @parameterized.parameters([
      {
          "start_shape": start_shape,
          "stop_shape": stop_shape,
          "dtype": dtype,
          "num": num
      }
      for start_shape in [(), (2,), (2, 2)]
      for stop_shape in [(), (2,), (2, 2)]
      for dtype in [np.float64, np.int64]
      for num in [0, 1, 2, 20]
  ])
  # pylint: enable=g-complex-comprehension
  def testLinspaceBroadcasts(self, start_shape, stop_shape, dtype, num):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the loop that does the real comparison is indented inside the branch that skips
    # the test for old numpy versions, and the skip call raises, so the comparison code is
    # effectively dead and never executes; the test either skips or passes without asserting.
    # Parameterized over start shape, stop shape (scalar, vector, matrix), dtype (float64 or int64)
    # and sample count (0, 1, 2, 20). Intended scenario: for each axis in the valid range, build
    # start as ones and stop as ten times ones, compute numpy linspace and the op under test along
    # that axis, evaluate it, and assert the two are close. Skips on numpy older than 1.16.
    if Version(np.version.version) < Version("1.16.0"):
      self.skipTest("numpy doesn't support axes before version 1.16.0")

      ndims = max(len(start_shape), len(stop_shape))
      for axis in range(-ndims, ndims):
        start = np.ones(start_shape, dtype)
        stop = 10 * np.ones(stop_shape, dtype)

        np_ans = np.linspace(start, stop, num, axis=axis)
        tf_ans = self.evaluate(
            math_ops.linspace_nd(start, stop, num, axis=axis))

        self.assertAllClose(np_ans, tf_ans)

  def testShapeInformationPeserved(self):
    # [FUNCTION DOC]
    # Scenario: wrap the op in a traced function, get a concrete function using partially unknown
    # float32 input specs of shape 64 by unknown, with constant count 10 and axis -1. Asserts the
    # inferred output shape is 64, unknown, 10, i.e. static shape information is preserved.
    # Takes no inputs, returns nothing.
    @def_function.function
    def linspace(start, stop, num, axis):
      # [FUNCTION DOC]
      # HELPER: traced wrapper used by the shape test. Calls the n-dimensional linspace with the
      # given start, stop, count and axis and returns its result tensor.
      return math_ops.linspace_nd(start, stop, num=num, axis=axis)

    # Constant num and axis leads to preserved known shape.
    output_shape = linspace.get_concrete_function(
        start=tensor.TensorSpec(shape=[64, None], dtype=dtypes.float32),
        stop=tensor.TensorSpec(shape=[64, None], dtype=dtypes.float32),
        num=10,
        axis=-1,
    ).output_shapes
    expected_shape = (64, None, 10)
    self.assertEqual(output_shape, expected_shape)


if __name__ == "__main__":
  googletest.main()
