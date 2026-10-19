# [MODULE DOC]
# Test module for complex number division on the XLA test device. It sets the XLA_FLAGS environment
# variable at import time so that CPU fast-math honors NaNs and infinities, then defines one test
# class that checks real_div on complex tensors for division by zero and for infinite/NaN
# numerators and denominators. Run as a script it starts the googletest main runner.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test cases for complex numbers division."""

import os

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.platform import googletest

os.environ["XLA_FLAGS"] = ("--xla_cpu_fast_math_honor_nans=true "
                           "--xla_cpu_fast_math_honor_infs=true")


class ComplexNumbersDivisionTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case covering complex division edge cases (zero, infinite and NaN operands).
  # Uses the XLA test device scope and the complex dtypes provided by the XLATestCase base class.
  """Test cases for complex numbers division operators."""

  def _testBinary(self, op, a, b, expected, equality_test=None):
    # [FUNCTION DOC]
    # HELPER: runs a binary op on the XLA test device for the complex division tests.
    # Inputs: op is a callable taking two tensors; a and b are numpy arrays of complex values;
    # expected is a numpy array of the expected complex results; equality_test is an optional
    # comparison callback (defaults to the type-aware approximate comparison).
    # Builds placeholders matching a and b inside the test scope, applies op, runs it in a session
    # and compares real and imaginary parts separately with a relative tolerance of 1e-3.
    # Returns nothing; assertion failures are raised by the comparison callback.
    with self.session() as session:
      with self.test_scope():
        pa = array_ops.placeholder(dtypes.as_dtype(a.dtype), a.shape, name="a")
        pb = array_ops.placeholder(dtypes.as_dtype(b.dtype), b.shape, name="b")
        output = op(pa, pb)
      result = session.run(output, {pa: a, pb: b})
      if equality_test is None:
        equality_test = self.assertAllCloseAccordingToType
      equality_test(np.real(result), np.real(expected), rtol=1e-3)
      equality_test(np.imag(result), np.imag(expected), rtol=1e-3)

  def testComplexOps(self):
    # [FUNCTION DOC]
    # For every complex dtype, checks real_div in three scenarios: (1) division by zero of
    # numerators including infinities and NaNs, expecting infinities or NaNs per component;
    # (2) finite numerator with infinite or NaN denominators, where some expected values
    # deliberately differ from Python's behavior (zeros); (3) infinite or NaN numerators over
    # finite denominators. Each check goes through the _testBinary helper. Returns nothing.
    for dtype in self.complex_types:
      # Test division by 0 scenarios.
      self._testBinary(
          gen_math_ops.real_div,
          np.array([
              complex(1, 1),
              complex(1, np.inf),
              complex(1, np.nan),
              complex(np.inf, 1),
              complex(np.inf, np.inf),
              complex(np.inf, np.nan),
              complex(np.nan, 1),
              complex(np.nan, np.inf),
              complex(np.nan, np.nan),
              complex(-np.inf, np.nan),
          ],
                   dtype=dtype),
          np.array([
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0.0 + 0j,
          ],
                   dtype=dtype),
          expected=np.array([
              complex(np.inf, np.inf),
              complex(np.inf, np.inf),
              complex(np.inf, np.nan),
              complex(np.inf, np.inf),
              complex(np.inf, np.inf),
              complex(np.inf, np.nan),
              complex(np.nan, np.inf),
              complex(np.nan, np.inf),
              complex(np.nan, np.nan),
              complex(-np.inf, np.nan),
          ],
                            dtype=dtype))

      # Test division with finite numerator, inf/nan denominator.
      self._testBinary(
          gen_math_ops.real_div,
          np.array([
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
          ],
                   dtype=dtype),
          np.array(
              [
                  complex(1, np.inf),
                  complex(1, np.nan),
                  complex(np.inf, 1),
                  complex(np.inf, np.inf),  # C++ and Python diverge here.
                  complex(np.inf, np.nan),  # C++ and Python diverge here.
                  complex(np.nan, 1),
                  complex(np.nan, np.inf),  # C++ and Python diverge here.
                  complex(np.nan, -np.inf),  # C++ and Python diverge here.
                  complex(np.nan, np.nan),
              ],
              dtype=dtype),
          expected=np.array(
              [
                  (1 + 1j) / complex(1, np.inf),
                  (1 + 1j) / complex(1, np.nan),
                  (1 + 1j) / complex(np.inf, 1),
                  complex(0 + 0j),  # C++ and Python diverge here.
                  complex(0 + 0j),  # C++ and Python diverge here.
                  (1 + 1j) / complex(np.nan, 1),
                  complex(0 + 0j),  # C++ and Python diverge here.
                  complex(0 - 0j),  # C++ and Python diverge here.
                  (1 + 1j) / complex(np.nan, np.nan),
              ],
              dtype=dtype))

      # Test division with inf/nan numerator, infinite denominator.
      self._testBinary(
          gen_math_ops.real_div,
          np.array([
              complex(1, np.inf),
              complex(1, np.nan),
              complex(np.inf, 1),
              complex(np.inf, np.inf),
              complex(np.inf, np.nan),
              complex(np.nan, 1),
              complex(np.nan, np.inf),
              complex(np.nan, np.nan),
              complex(np.nan, -np.inf),
          ],
                   dtype=dtype),
          np.array([
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              -1 - 1j,
          ],
                   dtype=dtype),
          expected=np.array(
              [
                  complex(np.inf, np.inf),  # C++ and Python diverge here.
                  complex(1 / np.nan) / (1 + 1j),
                  complex(np.inf / 1) / (1 + 1j),
                  complex(np.inf, -np.nan),  # C++ and Python diverge here.
                  complex(np.inf, -np.inf),  # C++ and Python diverge here.
                  complex(np.nan / 1) / (1 + 1j),
                  complex(np.inf, np.inf),  # C++ and Python diverge here.
                  complex(np.nan / np.nan) / (1 + 1j),
                  complex(np.inf, np.inf),  # C++ and Python diverge here.
              ],
              dtype=dtype))


if __name__ == "__main__":
  googletest.main()
