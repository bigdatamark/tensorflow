# [MODULE DOC]
# Tests for the einsum operation on XLA-backed devices. Defines one test class that compares
# einsum results (matrix product, implicit output form, reductions, transpose) with expected
# numpy values for each floating point type. No import-time side effects.
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
"""Test cases for einsum op."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.platform import googletest


class EinsumOpTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests einsum placed on the XLA test device, over all float types of the test base class.
  """Test cases for einsum op."""

  def _testUnary(self, op, inp, expected):
    # [FUNCTION DOC]
    # HELPER: runs a one-input operation on a placeholder fed with inp inside the test scope and
    # checks output dtype equals that of expected and values are close (type-dependent tolerance).
    """Verifies that unary 'op' produces 'expected' when fed input 'inp'."""
    with self.session() as session:
      with self.test_scope():
        pinp = array_ops.placeholder(
            dtypes.as_dtype(inp.dtype), inp.shape, name='a')
        output = op(pinp)
      result = session.run(output, {pinp: inp})
      self.assertEqual(output.dtype, expected.dtype)
      self.assertAllCloseAccordingToType(
          expected, result, rtol=1e-3, atol=1e-5, bfloat16_rtol=0.03)

  def _testBinary(self, op, a, b, expected):
    # [FUNCTION DOC]
    # HELPER: runs a two-input operation on placeholders fed with a and b inside the test scope and
    # asserts the result is close to expected.
    """Verifies that binary 'op' produces 'expected' when fed 'a' and 'b'."""
    with self.session() as session:
      with self.test_scope():
        pa = array_ops.placeholder(dtypes.as_dtype(a.dtype), a.shape, name='a')
        pb = array_ops.placeholder(dtypes.as_dtype(b.dtype), b.shape, name='b')
        output = op(pa, pb)
      result = session.run(output, {pa: a, pb: b})
      self.assertAllCloseAccordingToType(result, expected, rtol=1e-3)

  def testMatMul(self):
    # [FUNCTION DOC]
    # For each float type, einsum of a 1x1 and a 1x1 matrix with explicit output; expects -2.
    for dtype in self.float_types:
      self._testBinary(
          lambda x, y: special_math_ops.einsum('ij,jk->ik', x, y),
          np.array([[-0.25]], dtype=dtype),
          np.array([[8]], dtype=dtype),
          expected=np.array([[-2]], dtype=dtype))

  def testImplicitForm(self):
    # [FUNCTION DOC]
    # Einsum without an explicit output spec (implicit form) contracting all indices; expects 128.
    for dtype in self.float_types:
      self._testBinary(
          lambda x, y: special_math_ops.einsum('ijk,kji', x, y),
          np.array([[[1, 3], [2, 5], [6, 8]]], dtype=dtype),
          np.array([[[1], [3], [2]], [[5], [6], [8]]], dtype=dtype),
          expected=np.array(128, dtype=dtype))

  def testReducedIndices(self):
    # [FUNCTION DOC]
    # Einsum of a 3x2 matrix and a vector with an empty output (full reduction); expects 59.
    for dtype in self.float_types:
      self._testBinary(
          lambda x, y: special_math_ops.einsum('ij,j->', x, y),
          np.array([[1, 3], [2, 5], [6, 8]], dtype=dtype),
          np.array([3, 2], dtype=dtype),
          expected=np.array(59, dtype=dtype))

  def testUnary(self):
    # [FUNCTION DOC]
    # Unary einsum that reverses the axes of a 1x3x2 tensor; expects the transposed 2x3x1 tensor.
    for dtype in self.float_types:
      self._testUnary(
          lambda x: special_math_ops.einsum('ijk->kji', x),
          np.array([[[1, 3], [2, 5], [6, 8]]], dtype=dtype),
          expected=np.array([[[1], [2], [6]], [[3], [5], [8]]], dtype=dtype))


if __name__ == '__main__':
  googletest.main()
