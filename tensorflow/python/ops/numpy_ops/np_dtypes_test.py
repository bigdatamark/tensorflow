# [MODULE DOC]
# Unit tests for the dtype utilities in np_dtypes (default float type and result-type promotion under
# the prefer-float32 and allow-float64 switches). Run as a script it enables eager execution and
# starts the TensorFlow test runner. Test-only module; no registrations at import time.
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
"""Tests for tf-numpy dtype utilities."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import test


class DTypeTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case (parameterized) covering np_dtypes.default_float_type and np_dtypes._result_type for each
  # combination of the allow-float64 and prefer-float32 global switches. Holds no fixtures of its own.
  # KNOWN ISSUE: the tests set global module state and never restore it, so later tests in the same process see it.

  @parameterized.parameters([False, True])
  def testAllowF64False(self, prefer_f32):
    # [FUNCTION DOC]
    # KNOWN ISSUE: leaves allow-float64 disabled and prefer-float32 at the parameter value after the test; no cleanup.
    # Parameterized over prefer_f32 being false and true. Disables float64 and sets the prefer-float32
    # switch, then asserts the default float type is float32 and that the result type of a float64
    # zero-dimensional array combined with a Python float is float32 in both cases.
    np_dtypes.set_allow_float64(False)
    np_dtypes.set_prefer_float32(prefer_f32)
    self.assertEqual(dtypes.float32, np_dtypes.default_float_type())
    self.assertEqual(dtypes.float32,
                     np_dtypes._result_type(np.zeros([], np.float64), 1.1))

  def testAllowF64TruePreferF32False(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: leaves global switches changed (float64 allowed, prefer-float32 off) with no cleanup.
    # Allows float64 and turns prefer-float32 off. Asserts the default float type is float64, the result
    # type for a Python float is float64 and for a Python complex is complex128.
    np_dtypes.set_allow_float64(True)
    np_dtypes.set_prefer_float32(False)
    self.assertEqual(dtypes.float64, np_dtypes.default_float_type())
    self.assertEqual(dtypes.float64, np_dtypes._result_type(1.1))
    self.assertEqual(dtypes.complex128, np_dtypes._result_type(1.j))

  def testAllowF64TruePreferF32True(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: leaves global switches changed (prefer-float32 on) with no cleanup.
    # Allows float64 and turns prefer-float32 on. Asserts the default float type is float32; a Python
    # float gives float32; a float64 array with a Python float gives float64; a Python complex gives
    # complex64; a complex128 array with a Python complex gives complex128; a float32 array with a
    # Python complex gives complex64.
    np_dtypes.set_allow_float64(True)
    np_dtypes.set_prefer_float32(True)
    self.assertEqual(dtypes.float32, np_dtypes.default_float_type())
    self.assertEqual(dtypes.float32, np_dtypes._result_type(1.1))
    self.assertEqual(dtypes.float64,
                     np_dtypes._result_type(np.zeros([], np.float64), 1.1))
    self.assertEqual(dtypes.complex64, np_dtypes._result_type(1.1j))
    self.assertEqual(dtypes.complex128,
                     np_dtypes._result_type(np.zeros([], np.complex128), 1.1j))
    self.assertEqual(dtypes.complex64,
                     np_dtypes._result_type(np.zeros([], np.float32), 1.1j))


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
