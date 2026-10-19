# [MODULE DOC]
# Test module for compiling the Const op with XLA. Unlike other tests in this directory it does not
# use the XLA test case base class; it uses a function compiled with jit_compile enabled, because
# the XLA Const kernel is compile-only. Runs the standard test main when executed.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for const op compilation."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


# This test doesn't use XLATestCase like the other tests in this directory.
# The Const op xla op kernel is compilation only and therefore is not executed
# with XLA in the on demand compilation mode. Instead we use
# tf.function(jit_compile=True)
class ConstOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case checking that constants of many dtypes survive XLA jit compilation unchanged.

  # Verifies that the Const op works
  # @test_util.run_v2_only
  def testConst(self):
    # [FUNCTION DOC]
    # Iterates over bool, signed and unsigned integer, float16, bfloat16, float32, float64 and
    # several float8 dtypes (one sub-test each). Builds values: true/false for bool; zero, one,
    # dtype min and max for unsigned types (no negative one); additionally negative one for the
    # others; floating types also get positive infinity, negative infinity and NaN. Creates a
    # function compiled with XLA that returns a constant of those values, calls it, and asserts
    # the result equals the original array. Returns nothing.
    types = {
        dtypes.bool,
        dtypes.int8,
        dtypes.int16,
        dtypes.int32,
        dtypes.int64,
        dtypes.uint8,
        dtypes.uint16,
        dtypes.uint32,
        dtypes.uint64,
        dtypes.float16,
        dtypes.bfloat16,
        dtypes.float32,
        dtypes.float64,
        dtypes.float8_e5m2,
        dtypes.float8_e4m3fn,
        dtypes.float8_e4m3fnuz,
        dtypes.float8_e4m3b11fnuz,
        dtypes.float8_e5m2fnuz,
    }
    for dtype in types:
      with self.subTest(dtype=dtype):
        if dtype == dtypes.bool:
          values = [True, False]
        elif dtype in [
            dtypes.uint8,
            dtypes.uint16,
            dtypes.uint32,
            dtypes.uint64,
        ]:
          values = [0., 1., dtype.min, dtype.max]
        else:
          values = [0., 1., -1., dtype.min, dtype.max]
        if dtype.is_floating:
          values.extend([float("Inf"), -float("Inf"), float("NaN")])
        values = np.array(values, dtype=dtype.as_numpy_dtype)

        @def_function.function(jit_compile=True)
        def f():
          # [FUNCTION DOC]
          # Nested function compiled with XLA: takes no arguments and returns a constant tensor
          # built from the enclosing loop's values and dtype.
          return constant_op.constant(values, dtype)  # pylint: disable=cell-var-from-loop

        result = f()
        self.assertAllEqual(self.evaluate(result), values)


if __name__ == "__main__":
  test.main()
