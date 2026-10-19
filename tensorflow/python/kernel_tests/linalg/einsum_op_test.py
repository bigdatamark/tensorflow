# [MODULE DOC]
# Test and benchmark module for the Einstein summation op (the raw einsum op in the linalg ops).
# Forward tests compare the op with NumPy's einsum for unary, binary and higher arity equations,
# repeated labels, ellipsis, broadcasting, several element types, unknown-shape placeholders,
# empty tensors, invalid equations and output label repetition (which NumPy cannot do). Gradient
# tests use numerical gradient checking over real and complex types. A benchmark class times the
# new op against the older einsum implementation on CPU for a list of equations. TensorFloat-32
# is disabled for the test classes. Running the file as a script starts the TensorFlow test
# runner. No import-time side effects.
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
"""Tests for tensorflow.ops.Einsum."""

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


@test_util.run_all_without_tensor_float_32(
    'Tests einsum, which sometimes does a matmul with cuBLAS')
class EinsumOpTest(test.TestCase):
  # [CLASS DOC]
  # Forward correctness tests of the einsum op against NumPy. Decorated so that reduced-precision
  # TensorFloat-32 matrix multiplication is disabled. A shared helper generates seeded random
  # inputs for an equation and compares the results; individual tests list equations by category
  # (unary, binary, repeated labels, ellipsis, broadcasting, dtypes, placeholders, empty).

  def _check(self, s, *input_shapes, **kwargs):
    # [FUNCTION DOC]
    # HELPER: shared comparison used by most forward tests.
    # Inputs: s is the einsum equation string; input_shapes are the shapes (tuples) of the
    # operands, one per input; an optional keyword dtype (default single precision float) selects
    # the NumPy element type. Generates seeded (fixed seed) random operands, with an added random
    # imaginary part for complex types, runs NumPy's einsum and the TensorFlow op on the same
    # data, and asserts they are close (absolute and relative tolerance of one ten thousandth).
    # Returns nothing.
    dtype = kwargs.pop('dtype', np.float32)
    r = np.random.RandomState(0)
    inputs = []
    for shape in input_shapes:
      with self.subTest(s=s, shape=shape):
        arr = np.array(r.randn(*shape)).astype(dtype)
        if dtype == np.complex64 or dtype == np.complex128:
          arr += 1j * np.array(r.randn(*shape)).astype(dtype)
        inputs.append(arr)
    input_tensors = [constant_op.constant(x, shape=x.shape) for x in inputs]
    a = np.einsum(s, *inputs)
    b = self.evaluate(gen_linalg_ops.einsum(input_tensors, s))
    self.assertAllClose(a, b, atol=1e-4, rtol=1e-4)

  def testUnary(self):
    # [FUNCTION DOC]
    # Scenario: single-operand equations: scalar, full reduction, identity, selecting a single
    # axis sum and axis permutations. Each compared to NumPy by the shared helper.
    self._check('->', ())
    self._check('ab->', (3, 3))
    self._check('ab->ab', (3, 3))
    self._check('abc->b', (3, 4, 5))
    self._check('abc->ca', (3, 4, 5))
    self._check('abc->cab', (3, 4, 5))

  def testUnaryWithRepeatedLabels(self):
    # [FUNCTION DOC]
    # Scenario: single-operand equations where an input label repeats (traces and generalized
    # diagonals, with and without extra reduced axes). Each compared to NumPy via the helper.
    self._check('aa->', (3, 3))
    self._check('aa->a', (3, 3))
    self._check('aaa->', (3, 3, 3))
    self._check('aaa->a', (3, 3, 3))
    self._check('aab->a', (3, 3, 4))
    self._check('aabcc->a', (3, 3, 5, 4, 4))
    self._check('aabcc->ac', (3, 3, 5, 4, 4))
    self._check('aabcd->ad', (3, 3, 5, 4, 4))

  def testUnaryEllipsis(self):
    # [FUNCTION DOC]
    # Scenario: single-operand equations using an ellipsis: degenerate scalar cases, cases taken
    # from the dask project, ordinary transposes with varying numbers of leading dimensions, and
    # an ellipsis with a repeated label. Each compared to NumPy via the helper.
    # Unary cases with ellipsis.
    # Edge cases.
    self._check('...->...', ())
    self._check('...->', ())
    self._check('->...', ())

    # Tests from dask
    self._check('a...a->a...', (2, 2))
    self._check('a...a->', (2, 2))
    self._check('a...a->...', (2, 5, 1, 2))
    self._check('a...a->a...', (2, 1, 2))
    self._check('a...a->a...', (2, 3, 4, 5, 2))

    # Regular cases.
    self._check('...ijk->...ki', (3, 4, 5))
    self._check('...ijk->...ki', (1, 3, 4, 5))
    self._check('...ijk->...ki', (2, 2, 3, 4, 5))

    # Repeated indices.
    self._check('i...ii->...i', (3, 2, 3, 3))

  def testBinarySimple(self):
    # [FUNCTION DOC]
    # Scenario: two-operand equations that satisfy the restrictive form required by the
    # compiler (XLA) path: each label appears once in both inputs (batch or contraction) or once
    # in an input and the output (free). Covers scalars, dot products, matrix-vector, matrix
    # products, batched products and an outer-like case. Compared to NumPy via the helper.
    # Binary cases in XLA mode must have either (a) each index appearing exactly
    # once in both the inputs (batch or contraction index), or (b) appearing
    # exactly once in an input and in the output (free index).
    self._check(',->', (), ())
    self._check('a,a->', (3,), (3,))
    self._check('a,a->a', (3,), (3,))
    self._check('ab,b->a', (3, 4), (4,))
    self._check('ab,ab->', (3, 4), (3, 4))
    self._check('ab,bc->ac', (3, 4), (4, 5))
    self._check('nij,jk->nik', (5, 2, 3), (3, 4))
    self._check('abc,bad->abcd', (1, 2, 3), (2, 1, 4))
    # Based on https://github.com/google/jax/issues/37#issuecomment-448572187
    self._check('sa,shb->shab', (2, 1), (2, 3, 4))

  def testReducedIndices(self):
    # [FUNCTION DOC]
    # Scenario: two-operand equations where some labels appear in only one input and not in the
    # output, so they are summed out before contraction. Compared to NumPy via the helper. One
    # case repeats an equation already covered by the simple binary test.
    self._check('ba,b->', (3, 2), (3,))
    self._check('ab,ab->', (3, 4), (3, 4))
    self._check('abce,badf->abcd', (1, 2, 3, 4), (2, 1, 4, 3))

  def testRepeatedIndices(self):
    # [FUNCTION DOC]
    # Scenario: two-operand equations where a label repeats within one input (generalized
    # diagonal) combined with contraction or free axes, including cases from the dask project.
    # Compared to NumPy via the helper.
    # Repeated indices.
    self._check('ijj,k->ik', (2, 3, 3), (4,))
    self._check('aba,a->b', (3, 4, 3), (3,))
    # From https://github.com/dask/dask/pull/3412#discussion_r182413444
    self._check('aab,bc->ac', (2, 2, 3), (3, 4))
    self._check('aab,bcc->ac', (2, 2, 3), (3, 4, 4))

  def testEllipsis(self):
    # [FUNCTION DOC]
    # Scenario: two-operand equations using an ellipsis without broadcasting: batch matrix
    # multiply, empty batch dimensions, a contraction with transposition, and equations whose
    # output omits the ellipsis because the batch shape is empty. Compared to NumPy via the helper.
    # Batch matmul with ellipsis but without broadcasting.
    self._check('...mk,...kn->...mn', (5, 1, 2, 3), (5, 1, 3, 4))
    # Empty batch dimensions.
    self._check('...mk,...kn->...mn', (2, 3), (3, 4))
    # Tensor contraction with transpose.
    self._check('...ija,aijb...->ba...ij', (1, 2, 2, 3, 1), (1, 2, 3, 4, 1, 2))
    # Output subscripts may omit ellipsis when batch shape is empty.
    self._check('...mk,...kn->mn', (2, 3), (3, 4))
    self._check('...mk,kn->mn', (2, 3), (3, 4))
    self._check('mk,...kn->mn', (2, 3), (3, 4))

  def testBroadcasting(self):
    # [FUNCTION DOC]
    # Scenario: two-operand equations where the ellipsis dimensions of the operands differ in
    # rank or contain size one dimensions that broadcast, including ellipsis in the middle of
    # subscripts and two cases from a public forum answer. Compared to NumPy via the helper.
    # Batch matmul with broadcasting.
    self._check('...ij,...jk->...ik', (1, 2, 3), (3, 5))
    self._check('...ij,...jk->...ik', (2, 3), (1, 3, 5))
    self._check('...ij,...jk->...ik', (5, 2, 3), (3, 5))
    self._check('...ij,...jk->...ik', (2, 3), (5, 3, 5))
    self._check('...ij,...jk->...ik', (3, 1, 2, 3), (1, 1, 7, 3, 5))
    self._check('i...j,j...k->...ik', (2, 1, 3, 1, 3), (3, 1, 7, 5))
    # Following 2 from https://stackoverflow.com/a/19203475/1611416
    self._check('...abc,...abcd->...d', (1, 1, 2, 3, 4), (5, 2, 3, 4, 6))
    self._check('ab...,b->ab...', (2, 3, 1, 1, 5), (3,))
    self._check('i...j,j...k->i...k', (3, 1, 2, 2), (2, 2, 3, 1, 4))

  def testBroadcastingWithRepeatedIndices(self):
    # [FUNCTION DOC]
    # Scenario: equations combining ellipsis broadcasting with a repeated label (generalized
    # diagonal) in one operand. Compared to NumPy via the helper.
    # Broadcasting with repeated indices.
    self._check('ij,jk...k->i...', (3, 2), (2, 4, 1, 4))
    self._check('ij,jk...k->...i', (3, 2), (2, 4, 5, 4))
    self._check('ijj,jk...k->i...', (3, 2, 2), (2, 4, 1, 4))
    self._check('i...jj,jk...k->i...', (3, 3, 1, 2, 2), (2, 4, 1, 5, 4))

  def testDtypes(self):
    # [FUNCTION DOC]
    # Scenario: a two by two matrix product equation computed for bfloat16, single and double
    # precision floats, both complex types and 32 and 64 bit integers. For bfloat16 the NumPy
    # reference is computed in single precision then cast (NumPy lacks bfloat16 support) and a
    # looser tolerance (one hundredth) is used; other types use one ten thousandth.
    bfloat16 = dtypes.bfloat16.as_numpy_dtype

    def check(dtype):
      # [FUNCTION DOC]
      # HELPER: runs the matrix product comparison for one element type. Input: dtype, a NumPy
      # element type. Builds seeded random two by two operands (with random imaginary parts for
      # complex types), computes the NumPy reference, runs the op and asserts closeness within
      # the type-dependent tolerance. Returns nothing.
      r = np.random.RandomState(0)
      equation = 'ij,jk->ik'
      input_shapes = [(2, 2), (2, 2)]
      inputs = []
      for shape in input_shapes:
        with self.subTest(dtype=dtype, shape=shape):
          arr = np.array(r.randn(*shape)).astype(dtype)
          if dtype == np.complex64 or dtype == np.complex128:
            arr += 1j * np.array(r.randn(*shape)).astype(dtype)
          inputs.append(arr)
      input_tensors = [constant_op.constant(x) for x in inputs]
      if dtype == bfloat16:
        # np.einsum doesn't support bfloat16.
        a = np.einsum(equation,
                      *[x.astype(np.float32) for x in inputs]).astype(dtype)
      else:
        a = np.einsum(equation, *inputs)

      b = self.evaluate(gen_linalg_ops.einsum(input_tensors, equation))
      tol = 1e-2 if dtype == bfloat16 else 1e-4
      self.assertAllClose(a, b, atol=tol, rtol=tol)

    for dtype in [
        bfloat16, np.float32, np.float64, np.complex64, np.complex128, np.int32,
        np.int64
    ]:
      check(dtype)

  @test_util.disable_xla('b/131919749')
  @test_util.run_in_graph_and_eager_modes
  def testInvalid(self):
    # [FUNCTION DOC]
    # Scenario (skipped on XLA, runs in graph and eager modes): invalid equation and operand
    # combinations: wrong operand rank, inconsistent dimension sizes, incompatible broadcast
    # dimensions, and an output missing the ellipsis although the batch shape is non-empty. For
    # each, asserts a value error or invalid-argument error is raised both with fully known
    # shapes and when the operands are wrapped in unknown-shape placeholders.
    r = np.random.RandomState(0)
    cases = [
        # incorrect rank.
        ('ij,jk->ik', r.randn(1, 2, 3), r.randn(3, 4)),
        ('...ij,jk->ik', r.randn(3), r.randn(3, 4)),
        # inconsistent dimensions.
        ('ij,jk->ik', r.randn(2, 3), r.randn(4, 4)),
        # broadcasting is invalid
        ('...ij,...jk->...ik', r.randn(5, 2, 3), r.randn(7, 3, 4)),
        # output should have ellipsis when broadcasting shape is
        # non-empty.
        ('...ij,...jk->ik', r.randn(2, 2, 3), r.randn(3, 4)),
    ]
    for args in cases:
      with self.subTest(args=args):
        with self.assertRaises((ValueError, errors.InvalidArgumentError)):
          _ = self.evaluate(gen_linalg_ops.einsum(args[1:], args[0]))

        placeholders = [
            array_ops.placeholder_with_default(x, shape=None) for x in args[1:]
        ]
        with self.assertRaises((ValueError, errors.InvalidArgumentError)):
          _ = self.evaluate(gen_linalg_ops.einsum(placeholders, args[0]))

  @test_util.run_in_graph_and_eager_modes
  def testPlaceholder(self):
    # [FUNCTION DOC]
    # Scenario: operands wrapped as placeholders with a default value and partially or fully
    # unknown static shapes, for batch contraction and ellipsis equations. Asserts the results
    # match NumPy computed on the actual data.

    def check(equation, *input_and_placeholder_shapes):
      # [FUNCTION DOC]
      # HELPER: one placeholder comparison. Inputs: equation string; one pair per operand of
      # (actual shape, placeholder shape, which may contain unknown dimensions or be none).
      # Generates seeded random operands, feeds them as defaults of placeholders, computes the
      # NumPy reference, runs the op and asserts closeness. Returns nothing.
      r = np.random.RandomState(0)
      inputs = []
      input_placeholders = []
      for actual_shape, placeholder_shape in input_and_placeholder_shapes:
        with self.subTest(equation=equation, actual_shape=actual_shape,
                          placeholder_shape=placeholder_shape):
          input_np = np.array(r.randn(*actual_shape))
          inputs.append(input_np)
          input_placeholders.append(
              array_ops.placeholder_with_default(input_np, placeholder_shape))

      a = np.einsum(equation, *inputs)
      b = self.evaluate(gen_linalg_ops.einsum(input_placeholders, equation))
      self.assertAllClose(a, b, atol=1e-4, rtol=1e-4)

    check('bijl,bjkm->bik', ((9, 2, 3, 5), (None, None, None, 5)),
          ((9, 3, 4, 7), (None, None, 4, None)))
    check('bijl,bjkm->bik', ((9, 2, 3, 5), None), ((9, 3, 4, 7), None))
    check('...ij,...->...i', ((4, 3, 1, 2), (None, 3, None, 2)),
          ((4, 3), (None, 3)))
    check('...ij,...jk->...ik', ((3, 1, 2, 3), None), ((1, 7, 3, 4), None))

  def testOutputRepeatedLabels(self):
    # [FUNCTION DOC]
    # Scenario: an equation whose output repeats a label (building a diagonal matrix from a
    # vector), the reverse of a trace, which NumPy does not support and which gradients need.
    # The expected answer is built by hand from the diagonal of a random two by two matrix.
    # Asserts the op result is close to that diagonal matrix.
    # This is the reverse operation of generalized traces, to be used for
    # computing symbolic gradients of einsum. Note: this operation is not
    # supported by np.einsum as it's only required for gradients.
    r = np.random.RandomState(0)
    a = r.randn(2, 2)
    s = 'a->aa'
    diag_a = np.diag(np.diag(a))
    b = self.evaluate(gen_linalg_ops.einsum([np.diag(a)], s))
    self.assertAllClose(diag_a, b, atol=1e-4, rtol=1e-4)

  def testEmpty(self):
    # [FUNCTION DOC]
    # Scenario: contractions over zero-sized dimensions (including a shape from a transformer
    # model), whose results must be zero-filled tensors of the expected shape. NumPy is not
    # used; the output is compared with zeros.
    def check(equation, input_shapes, output_shape):
      # [FUNCTION DOC]
      # HELPER: one empty-input check. Inputs: equation string; list of operand shapes; the
      # expected output shape. Builds seeded random operands, runs the op and asserts the result
      # is close to zeros of the expected shape. Returns nothing.
      # All these cases result in an output filled with zeros, so we don't call
      # np.einsum. Also np.einsum doesn't support generalized diagonals which
      # are needed for EinsumOp gradients.
      r = np.random.RandomState(0)
      inputs = [np.array(r.randn(*shape)) for shape in input_shapes]
      output = self.evaluate(gen_linalg_ops.einsum(inputs, equation))
      self.assertAllClose(output, np.zeros(output_shape), atol=1e-4, rtol=1e-4)

    # Contractions along zero-sized dimensions.
    check('ab,bc->ac', [(0, 10), (10, 10)], (0, 10))
    # From transformer xl.
    check('ibnd,ijbn->jnd', [(1, 0, 5, 10), (1, 1, 0, 5)], (1, 5, 10))

  def testEmptyWithRepeatedLabels(self):
    # [FUNCTION DOC]
    # Scenario: zero-sized operands in equations with repeated labels (generalized traces and
    # diagonals, including repeated output labels) combined with contraction. The result must
    # be a zero tensor of the expected shape; NumPy is not used because it cannot compute these.

    def check(equation, input_shapes, output_shape):
      # [FUNCTION DOC]
      # HELPER: one empty-input check with repeated labels. Inputs: equation string; list of
      # operand shapes; expected output shape. Builds seeded random operands, runs the op and
      # asserts the result is close to zeros of the expected shape. Returns nothing.
      # All these cases result in an output filled with zeros, so we don't call
      # np.einsum. Also np.einsum doesn't support generalized diagonals which
      # are needed for EinsumOp gradients.
      r = np.random.RandomState(0)
      inputs = [np.array(r.randn(*shape)) for shape in input_shapes]
      output = self.evaluate(gen_linalg_ops.einsum(inputs, equation))
      self.assertAllClose(output, np.zeros(output_shape), atol=1e-4, rtol=1e-4)

    # Generalized traces with zero-sized dimensions.
    check('aab,bc->ac', [(0, 0, 10), (10, 10)], (0, 10))
    check('aaab,bc->c', [(0, 0, 0, 3), (3, 4)], (4,))
    # Generalized diagonals along with contraction.
    check('ab,bc->aaca', [(0, 10), (10, 5)], (0, 0, 5, 0))
    check('ab,bc->aaa', [(0, 10), (10, 5)], (0, 0, 0))
    check('ab,bc->cc', [(0, 10), (10, 5)], (5, 5))
    check('ab,ab->aaa', [(0, 5), (0, 5)], (0, 0, 0))


@test_util.run_all_in_graph_and_eager_modes
@test_util.run_all_without_tensor_float_32(
    "Tests einsum's gradient, which sometimes does a matmul with cuBLAS")
class EinsumGradTest(test.TestCase):
  # [CLASS DOC]
  # Gradient tests of the einsum op using numerical gradient checking. Decorated to run every
  # test in both graph and eager modes with TensorFloat-32 disabled. A shared helper loops over
  # real and complex single and double precision types for each equation.

  def _check_gradient(self, s, *input_shapes):
    # [FUNCTION DOC]
    # HELPER: shared gradient check for the gradient tests.
    # Inputs: s is the einsum equation string; input_shapes are operand shapes (tuples). In a
    # cached session, for each of four element types (float32, float64, complex64, complex128),
    # generates seeded random operands (complex ones get a random imaginary part), computes the
    # analytical and numerical gradients of the op with respect to all operands, and asserts the
    # maximum error is below a tolerance of ten times the square root of the type's decimal
    # resolution. Each type runs as a sub-test. Returns nothing.
    with self.cached_session():
      r = np.random.RandomState(seed=0)
      for dtype in (np.float32, np.float64, np.complex64, np.complex128):
        with self.subTest(s=s, dtype=dtype):
          tol = 10 * np.sqrt(np.finfo(dtype).resolution)
          if dtype in (np.complex64, np.complex128):
            inputs = [
                np.array(r.randn(*shape), dtype) +
                1j * np.array(r.randn(*shape), dtype) for shape in input_shapes
            ]
          else:
            inputs = [
                np.array(r.randn(*shape), dtype) for shape in input_shapes]
          input_tensors = [
              constant_op.constant(x, shape=x.shape) for x in inputs]
          analytical, numerical = gradient_checker_v2.compute_gradient(
              lambda *xs: gen_linalg_ops.einsum(xs, s), input_tensors)
          self.assertLess(
              gradient_checker_v2.max_error(analytical, numerical), tol)

  def testUnary(self):
    # [FUNCTION DOC]
    # Scenario: gradient check of single-operand equations (scalar, trace-like diagonal
    # extraction, repeated labels with a repeated output label, and a transpose).
    # Unary cases.
    self._check_gradient('->', ())
    self._check_gradient('aaa->a', (3, 3, 3))
    self._check_gradient('aabcd->ad', (3, 3, 5, 4, 4))
    self._check_gradient('aabcd->add', (3, 3, 5, 4, 4))
    self._check_gradient('abcd->da', (3, 5, 4, 2))

  def testUnaryEllipsis(self):
    # [FUNCTION DOC]
    # Scenario: gradient check of single-operand equations with ellipsis: scalar cases, dask
    # derived cases, ordinary transposes and an ellipsis in the middle of the subscripts.
    self._check_gradient('...->...', ())
    self._check_gradient('...->', ())
    self._check_gradient('->...', ())

    # Tests from dask
    self._check_gradient('a...a->a...', (2, 2))
    self._check_gradient('a...a->', (2, 2))
    self._check_gradient('a...a->...', (2, 5, 1, 2))
    self._check_gradient('a...a->a...', (2, 1, 2))
    self._check_gradient('a...a->a...', (2, 3, 4, 5, 2))

    self._check_gradient('...ijk->...ki', (3, 4, 5))
    self._check_gradient('...ijk->...ki', (1, 3, 4, 5))
    self._check_gradient('...ijk->...ki', (2, 2, 3, 4, 5))
    self._check_gradient('ab...cd->da...', (3, 5, 2, 3, 4, 2))

  def testBinarySimple(self):
    # [FUNCTION DOC]
    # Scenario: gradient check of the simple two-operand equations (dot products, matrix-vector,
    # matrix and batch products, one outer-like case) in the form required by the XLA path.
    # Binary cases in XLA mode must have either (a) each index appearing exactly
    # once in both the inputs (batch or contraction index), or (b) appearing
    # exactly once in an input and in the output (free index).
    self._check_gradient(',->', (), ())
    self._check_gradient('a,a->', (3,), (3,))
    self._check_gradient('a,a->a', (3,), (3,))
    self._check_gradient('ab,b->a', (3, 4), (4,))
    self._check_gradient('ab,ab->', (3, 4), (3, 4))
    self._check_gradient('ab,bc->ac', (3, 4), (4, 5))
    self._check_gradient('nij,jk->nik', (5, 2, 3), (3, 4))
    self._check_gradient('abc,bad->abcd', (1, 2, 3), (2, 1, 4))
    # Based on https://github.com/google/jax/issues/37#issuecomment-448572187
    self._check_gradient('sa,shb->shab', (2, 1), (2, 3, 4))

  def testEmpty(self):
    # [FUNCTION DOC]
    # Scenario: gradient check of a contraction over zero-sized dimensions taken from a
    # transformer model.
    # From Transformer XL.
    self._check_gradient('ibnd,ijbn->jnd', (1, 0, 5, 10), (1, 1, 0, 5))

  def testReducedIndices(self):
    # [FUNCTION DOC]
    # Scenario: gradient check of two-operand equations where some labels are summed out of a
    # single input before contraction.
    self._check_gradient('ba,b->', (3, 2), (3,))
    self._check_gradient('ab,ab->', (3, 4), (3, 4))
    self._check_gradient('ijkm,ijln->ijmn', (2, 3, 3, 4), (2, 3, 3, 2))
    self._check_gradient('abce,badf->abcd', (1, 2, 3, 4), (2, 1, 4, 3))

  def testReducedIndicesWithRepeatedLabels(self):
    # [FUNCTION DOC]
    # Scenario: gradient check of an equation combining reduced labels with a repeated output
    # label (the output builds a generalized diagonal).
    self._check_gradient('abce,badf->bcba', (1, 2, 3, 4), (2, 1, 4, 3))

  def testRepeatedLabels(self):
    # [FUNCTION DOC]
    # Scenario: gradient check of two-operand equations with a label repeated within one input,
    # including cases from the dask project.
    # Repeated indices.
    self._check_gradient('aba,a->b', (3, 4, 3), (3,))
    self._check_gradient('ijj,k->ik', (2, 3, 3), (4,))
    self._check_gradient('ill,k->ik', (2, 3, 3), (4,))
    # From https://github.com/dask/dask/pull/3412#discussion_r182413444
    self._check_gradient('aab,bc->ac', (1, 1, 3), (3, 4))
    self._check_gradient('aab,bcc->ac', (2, 2, 3), (3, 4, 4))

  def testEmptyWithRepeatedLabels(self):
    # [FUNCTION DOC]
    # Scenario: gradient check of equations with repeated labels where some dimensions have
    # size zero.
    self._check_gradient('aab,bc->ac', (0, 0, 10), (10, 10))
    self._check_gradient('aab,bc->ac', (1, 1, 0), (0, 10))
    self._check_gradient('aaab,bc->c', (0, 0, 0, 3), (3, 4))

  def testBroadcasting(self):
    # [FUNCTION DOC]
    # Scenario: gradient check of two-operand equations with ellipsis dimensions that broadcast.
    self._check_gradient('...ij,...jk->...ik', (3, 2), (2, 4))
    self._check_gradient('ij...,jk...->ik...', (3, 2, 1), (2, 4))
    self._check_gradient('...ij,...jk->...ik', (3, 1, 3, 2), (1, 5, 2, 4))
    self._check_gradient('i...j,j...k->i...k', (3, 1, 2, 2), (2, 2, 3, 1, 4))

  def testBroadcastingWithRepeatedLabels(self):
    # [FUNCTION DOC]
    # Scenario: gradient check of equations combining ellipsis broadcasting with a repeated
    # label in one operand.
    self._check_gradient('ij,jk...k->i...', (3, 2), (2, 4, 1, 4))
    self._check_gradient('aab,b...c->a...c', (1, 1, 3), (3, 1, 1, 4))


class EinsumBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark comparing the older einsum implementation with the new raw einsum op on CPU. The
  # class attribute lists equations paired with the size used for every dimension.
  cases = [
      # Unary cases.
      ['ijk->i', 100],
      ['ijk->kji', 100],
      # Regular matmul or batch matmul.
      ['ij,jk->ik', 1000],
      ['ji,kj->ik', 1000],
      ['ab,ab->', 100],
      ['ab,ba->', 100],
      ['abc,abc->', 100],
      ['abc,bac->', 100],
      ['abc,cba->', 100],
      ['bij,bjk->bik', 100],
      ['bji,bjk->bki', 100],
      ['ikl,kji->kl', 100],
      ['klj,lki->ij', 100],
      ['ijk,ilj->kli', 100],
      ['kij,mkb->ijmb', 100],
      ['abcd,ad->bc', 40],
      # Larger binary contractions.
      ['ijk,jklm->il', 40],
      ['efabc,eabcd->efd', 30],
      ['fabec,abcde->fde', 30],
      ['efabc,edabc->efd', 30],
      ['eadbf,dfebc->ecfad', 30],
      ['abcdef,bcdfg->abcdeg', 30],
  ]

  def benchmarkEinsum(self):
    # [FUNCTION DOC]
    # Benchmark: for each configured equation and dimension size, builds a CPU graph with
    # seeded random single precision variables (every dimension of every operand has that size),
    # initializes them, then times the older einsum function and the new raw op for at least
    # fifty iterations each, with names containing the equation and size. Reports results
    # through the benchmark framework; returns nothing.
    for equation, dim in self.cases:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device('/cpu:0'):
        r = np.random.RandomState(0)
        input_subscripts = equation.split('->')[0].split(',')
        input_vars = []
        for subscript in input_subscripts:
          input_shape = (dim,) * len(subscript)
          input_vars.append(
              variables.Variable(np.array(r.randn(*input_shape), np.float32)))
        self.evaluate(variables.global_variables_initializer())

        # Call einsum_v1.
        self.run_op_benchmark(
            sess,
            special_math_ops.einsum(equation, *input_vars),
            min_iters=50,
            name='einsum_v1_cpu_({})_{}'.format(equation, dim))

        # Call gen_linalg_ops.einsum.
        self.run_op_benchmark(
            sess,
            gen_linalg_ops.einsum(input_vars, equation),
            min_iters=50,
            name='einsum_v2_cpu_({})_{}'.format(equation, dim))


if __name__ == '__main__':
  test.main()
