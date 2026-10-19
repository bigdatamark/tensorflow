# [MODULE DOC]
# Test module for the cast operation under XLA. Checks casts between boolean, integer, float,
# complex and (separately) 8-bit float types across several shapes, including empty shapes and
# special values, by comparing XLA output against numpy conversions. No import-time side effects.
# Copyright 2025 The OpenXLA Authors.
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
import platform

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class CastTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for dtype casting. Provides a shared checker that casts numpy data on the XLA
  # device and compares with numpy conversion. Holds no state of its own.

  def test_cast(self):
    # [FUNCTION DOC]
    # Runs the cast checker for every ordered pair of source and destination types drawn from
    # bool, float32, float64, complex64, int32, int64, uint32 and uint64 (including identity
    # casts) in one shared session. Returns nothing.
    types = {
        dtypes.bool,
        dtypes.float32,
        dtypes.float64,
        dtypes.complex64,
        dtypes.int32,
        dtypes.int64,
        dtypes.uint32,
        dtypes.uint64,
    }
    with self.session() as session:
      for src_type in types:
        for dst_type in types:
          self._test_cast(src_type, dst_type, session)

  def test_cast_fp8(self):
    # [FUNCTION DOC]
    # Runs the cast checker between each of five 8-bit float formats and every other type in the
    # list of ordinary types plus the 8-bit formats, in both directions. Skipped on macOS because
    # casts to 8-bit floats occasionally yield NaN there. Returns nothing.
    if platform.system() == "Darwin":
      # TODO(b/271327511): Fix issue where casts to FP8 very rarely result in
      # NaN on Mac
      self.skipTest("Casts to FP8 sometimes result in NaN on Mac")
    fp8_types = {
        dtypes.float8_e5m2,
        dtypes.float8_e4m3fn,
        dtypes.float8_e4m3fnuz,
        dtypes.float8_e4m3b11fnuz,
        dtypes.float8_e5m2fnuz,
    }
    other_types = {
        dtypes.bool,
        dtypes.float32,
        dtypes.float64,
        dtypes.complex64,
        dtypes.int32,
        dtypes.int64,
        dtypes.uint32,
        dtypes.uint64,
    }
    with self.session() as session:
      for fp8_type in fp8_types:
        for other_type in other_types | fp8_types:
          self._test_cast(fp8_type, other_type, session)
          self._test_cast(other_type, fp8_type, session)

  def _test_cast(self, src_type, dst_type, session):
    # [FUNCTION DOC]
    # HELPER: shared checker for the cast tests.
    # For a source and destination dtype and a session, runs inside a sub-test labelled with the
    # types. First, for shapes scalar, length 4, 2x3 and 2x0x4, builds sequential values in the
    # source type (adding an imaginary part for complex sources), computes the expected result
    # with numpy conversion, and asserts the XLA cast output matches. Then it builds special
    # values: integer extremes for integer sources, integer-range extremes for float to integer
    # casts (maximum halved), or float min, max, NaN, epsilon, tiny and infinities for float to
    # float casts; then checks the same way. For other source types the previous data is reused.
    # Returns nothing.
    with self.subTest(src_type=src_type, dst_type=dst_type):
      shapes = [[], [4], [2, 3], [2, 0, 4]]
      src_np_dtype = src_type.as_numpy_dtype
      dst_np_dtype = dst_type.as_numpy_dtype

      for shape in shapes:
        src = np.arange(np.prod(shape)).astype(src_np_dtype)

        if src_type in self.complex_tf_types:
          src += (np.arange(np.prod(shape)) * 2j).astype(src_np_dtype)
        src = src.reshape(shape)
        dst = src.astype(dst_np_dtype)
        self.assert_op_output_matches_expected(
            lambda x, dst_type=dst_type: math_ops.cast(x, dst_type),
            src,
            expected=dst,
            local_session=session,
        )

      # Check special values.
      if src_type.is_integer:
        imin = np.iinfo(src_np_dtype).min
        imax = np.iinfo(src_np_dtype).max
        if src_type.is_unsigned:
          src = np.array([imin, imax, 0, 1], dtype=src_np_dtype)
        else:
          src = np.array([imin, imax, 0, 1, -1], dtype=src_np_dtype)
      elif src_type in self.float_tf_types:
        if dst_type.is_integer:
          imin = np.iinfo(dst_np_dtype).min
          imax = np.iinfo(dst_np_dtype).max // 2
          src = np.array([imin, imax, 0, 1], dtype=src_np_dtype)
        elif dst_type in self.float_tf_types:
          fmin = np.finfo(dst_np_dtype).min
          fmax = np.finfo(dst_np_dtype).max
          tiny = np.finfo(dst_np_dtype).tiny
          eps = np.finfo(dst_np_dtype).eps
          src = np.array(
              [fmin, fmax, np.nan, eps, -eps, tiny, -tiny, np.inf, -np.inf],
              dtype=src_np_dtype,
          )
      dst = src.astype(dst_np_dtype)
      self.assert_op_output_matches_expected(
          lambda x, dst_type=dst_type: math_ops.cast(x, dst_type),
          src,
          expected=dst,
          local_session=session,
      )

  def test_give_me_a_name(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: placeholder test with a meaningless name and an empty body; tests nothing.
    # Always passes. Returns nothing.
    pass


if __name__ == "__main__":
  googletest.main()
