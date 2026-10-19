# [MODULE DOC]
# Tests for the circulant linear operators (one, two and three dimensional, the latter two being
# block-circulant) and for the exponential power convolution kernel generator. Contains a test
# case for the kernel generator, shared base mixins that hold tolerances and an explicit (slow,
# loop-based) construction of the dense circulant matrix from a spectrum, derived-class test cases
# that plug into the shared linear operator test utility for self-adjoint, Hermitian and
# non-Hermitian spectra in 1D and 2D, and a simpler direct test case for 3D. Import-time side
# effects: creates a seeded module-level random number generator and module-level aliases to
# private helpers of the circulant operator module. When run as a script it registers the
# generated tests and starts the test runner.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
import contextlib

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.platform import test

rng = np.random.RandomState(0)
_to_complex = linear_operator_circulant._to_complex

exponential_power_convolution_kernel = (
    linear_operator_circulant.exponential_power_convolution_kernel)


def _operator_from_kernel(kernel, d, **kwargs):
  # [FUNCTION DOC]
  # HELPER: builds circulant operators from convolution kernels for the kernel tests.
  # Takes a convolution kernel tensor, the number of spatial dimensions d (1, 2 or 3) and extra
  # keyword arguments forwarded to the operator constructor. Casts the kernel to complex64, applies
  # the d-dimensional FFT to get the spectrum, and returns the circulant operator of matching
  # dimensionality (plain, 2D block or 3D block). If d is anything else, no branch matches and
  # nothing (null) is returned silently.
  spectrum = linear_operator_circulant._FFT_OP[d](
      math_ops.cast(kernel, dtypes.complex64))
  if d == 1:
    return linear_operator_circulant.LinearOperatorCirculant(spectrum, **kwargs)
  elif d == 2:
    return linear_operator_circulant.LinearOperatorCirculant2D(
        spectrum, **kwargs)
  elif d == 3:
    return linear_operator_circulant.LinearOperatorCirculant3D(
        spectrum, **kwargs)


def _spectrum_for_symmetric_circulant(
    spectrum_shape,
    d,
    ensure_self_adjoint_and_pd,
    dtype,
):
  # [FUNCTION DOC]
  # HELPER: builds spectra for the symmetric circulant operator tests.
  # Inputs: spectrum_shape (full shape of the spectrum, last d entries are the grid), d (number of
  # spatial dimensions), ensure_self_adjoint_and_pd (if true use a power-one kernel with zero
  # inflation, which yields a positive spectrum; otherwise power two with no inflation, which
  # yields some negative spectral values) and dtype (target dtype). Builds an exponential power
  # convolution kernel on the grid (an empty kernel for an empty grid), takes its FFT, casts to
  # dtype and broadcasts to the full spectrum shape. The kernel is real and symmetric so the
  # spectrum is real. Returns the spectrum tensor.
  """Spectrum for d-dimensional real/symmetric circulant."""
  grid_shape = spectrum_shape[-d:]

  if grid_shape == (0,) * d:
    kernel = array_ops.reshape(math_ops.cast([], dtype), grid_shape)
  else:
    kernel = exponential_power_convolution_kernel(
        grid_shape=grid_shape,
        # power=2 with this scale and no inflation will have some negative
        # spectra. It will still be real/symmetric.
        length_scale=math_ops.cast([0.2] * d, dtype.real_dtype),
        power=1 if ensure_self_adjoint_and_pd else 2,
        zero_inflation=0.2 if ensure_self_adjoint_and_pd else None,
    )
  spectrum = linear_operator_circulant._FFT_OP[d](_to_complex(kernel))
  spectrum = math_ops.cast(spectrum, dtype)
  return array_ops.broadcast_to(spectrum, spectrum_shape)


@test_util.run_all_in_graph_and_eager_modes
class ExponentialPowerConvolutionKernelTest(parameterized.TestCase,
                                            test.TestCase):
  # [CLASS DOC]
  # Parameterized tests for the exponential power convolution kernel generator. Each test builds
  # a kernel, turns it into a circulant operator of matching dimension, densifies it and checks
  # properties of the resulting matrix (real, symmetric, unit diagonal, effect of zero inflation,
  # limiting cases of tiny and huge length scale). Runs in graph and eager modes.

  def assert_diag_is_ones(self, matrix, rtol):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the kernel tests. Asserts the diagonal of the given matrix is
    # all ones within the given relative tolerance; fails the test otherwise. Returns nothing.
    self.assertAllClose(
        np.ones_like(np.diag(matrix)), np.diag(matrix), rtol=rtol)

  def assert_real_symmetric(self, matrix, tol):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the kernel tests. Asserts the imaginary part of the (complex)
    # matrix is zero within the absolute tolerance, and that the real part equals its transpose
    # within the relative tolerance (both using the single tol value). Returns nothing; fails the
    # test otherwise.
    self.assertAllClose(np.zeros_like(matrix.imag), matrix.imag, atol=tol)
    self.assertAllClose(matrix.real, matrix.real.T, rtol=tol)

  @parameterized.named_parameters(
      dict(testcase_name="1Deven_power1", grid_shape=[10], power=1.),
      dict(testcase_name="2Deven_power1", grid_shape=[4, 6], power=1.),
      dict(testcase_name="3Deven_power1", grid_shape=[4, 6, 8], power=1.),
      dict(testcase_name="3Devenodd_power1", grid_shape=[4, 5, 7], power=1.),
      dict(testcase_name="1Dodd_power2", grid_shape=[9], power=2.),
      dict(testcase_name="2Deven_power2", grid_shape=[8, 4], power=2.),
      dict(testcase_name="3Devenodd_power2", grid_shape=[4, 5, 3], power=2.),
  )
  def test_makes_symmetric_and_real_circulant_with_ones_diag(
      self, grid_shape, power):
    # [FUNCTION DOC]
    # Parameterized over 1D, 2D and 3D grids (even and odd sizes) and power one or two. Builds a
    # kernel with a fixed length scale of 0.2 per dimension, turns it into a circulant operator,
    # densifies it, and asserts the matrix is real, symmetric and has an all-ones diagonal, with a
    # tolerance proportional to machine epsilon times the grid size.
    d = len(grid_shape)
    length_scale = [0.2] * d
    kernel = exponential_power_convolution_kernel(
        grid_shape=grid_shape,
        length_scale=length_scale,
        power=power)
    operator = _operator_from_kernel(kernel, d)

    matrix = self.evaluate(operator.to_dense())

    tol = np.finfo(matrix.dtype).eps * np.prod(grid_shape)
    self.assert_real_symmetric(matrix, tol)
    self.assert_diag_is_ones(matrix, rtol=tol)

  @parameterized.named_parameters(
      dict(testcase_name="1D", grid_shape=[10]),
      dict(testcase_name="2D", grid_shape=[5, 5]),
      dict(testcase_name="3D", grid_shape=[5, 4, 3]),
  )
  def test_zero_inflation(self, grid_shape):
    # [FUNCTION DOC]
    # Parameterized over 1D, 2D and 3D grids. Builds kernels with zero inflation unset, 0.5 and
    # 1.0, densifies the corresponding operators and asserts: all matrices are real symmetric with
    # unit diagonal; with inflation 0.5 the off-diagonal entries are half those of the
    # uninflated matrix; with inflation 1.0 the off-diagonal entries are all zero (identity).
    d = len(grid_shape)
    length_scale = [0.2] * d

    kernel_no_inflation = exponential_power_convolution_kernel(
        grid_shape=grid_shape,
        length_scale=length_scale,
        zero_inflation=None,
    )
    matrix_no_inflation = self.evaluate(
        _operator_from_kernel(kernel_no_inflation, d).to_dense())

    kernel_inflation_one_half = exponential_power_convolution_kernel(
        grid_shape=grid_shape,
        length_scale=length_scale,
        zero_inflation=0.5,
    )
    matrix_inflation_one_half = self.evaluate(
        _operator_from_kernel(kernel_inflation_one_half, d).to_dense())

    kernel_inflation_one = exponential_power_convolution_kernel(
        grid_shape=grid_shape,
        length_scale=length_scale,
        zero_inflation=1.0,
    )
    matrix_inflation_one = self.evaluate(
        _operator_from_kernel(kernel_inflation_one, d).to_dense())

    tol = np.finfo(matrix_no_inflation.dtype).eps * np.prod(grid_shape)

    # In all cases, matrix should be real and symmetric.
    self.assert_real_symmetric(matrix_no_inflation, tol)
    self.assert_real_symmetric(matrix_inflation_one, tol)
    self.assert_real_symmetric(matrix_inflation_one_half, tol)

    # In all cases, the diagonal should be all ones.
    self.assert_diag_is_ones(matrix_no_inflation, rtol=tol)
    self.assert_diag_is_ones(matrix_inflation_one_half, rtol=tol)
    self.assert_diag_is_ones(matrix_inflation_one, rtol=tol)

    def _matrix_with_zerod_diag(matrix):
      # [FUNCTION DOC]
      # HELPER: local helper for the zero inflation test. Takes a square matrix and returns a
      # copy with its diagonal set to zero (off-diagonal entries unchanged); no side effects.
      return matrix - np.diag(np.diag(matrix))

    # Inflation = 0.5 means the off-diagonal is deflated by factor (1 - .5) = .5
    self.assertAllClose(
        _matrix_with_zerod_diag(matrix_no_inflation) * 0.5,
        _matrix_with_zerod_diag(matrix_inflation_one_half), rtol=tol)

    # Inflation = 1.0 means the off-diagonal is deflated by factor (1 - 1) = 0
    self.assertAllClose(
        np.zeros_like(matrix_inflation_one),
        _matrix_with_zerod_diag(matrix_inflation_one), rtol=tol)

  @parameterized.named_parameters(
      dict(testcase_name="1D", grid_shape=[10]),
      dict(testcase_name="2D", grid_shape=[5, 5]),
      dict(testcase_name="3D", grid_shape=[5, 4, 3]),
  )
  def test_tiny_scale_corresponds_to_identity_matrix(self, grid_shape):
    # [FUNCTION DOC]
    # Parameterized over 1D, 2D and 3D grids. Builds a power-two kernel with a very small length
    # scale (0.001), densifies the operator, and asserts it is the identity matrix (within a
    # tolerance based on machine epsilon and grid size) and is real symmetric.
    d = len(grid_shape)

    kernel = exponential_power_convolution_kernel(
        grid_shape=grid_shape, length_scale=[0.001] * d, power=2)
    matrix = self.evaluate(_operator_from_kernel(kernel, d).to_dense())

    tol = np.finfo(matrix.dtype).eps * np.prod(grid_shape)
    self.assertAllClose(matrix, np.eye(np.prod(grid_shape)), atol=tol)
    self.assert_real_symmetric(matrix, tol)

  @parameterized.named_parameters(
      dict(testcase_name="1D", grid_shape=[10]),
      dict(testcase_name="2D", grid_shape=[5, 5]),
      dict(testcase_name="3D", grid_shape=[5, 4, 3]),
  )
  def test_huge_scale_corresponds_to_ones_matrix(self, grid_shape):
    # [FUNCTION DOC]
    # Parameterized over 1D, 2D and 3D grids. Builds a power-two kernel with a very large length
    # scale (100), densifies the operator, and asserts it is real symmetric and equal to the
    # all-ones matrix, using a tolerance of machine epsilon times grid size times fifty.
    d = len(grid_shape)

    kernel = exponential_power_convolution_kernel(
        grid_shape=grid_shape, length_scale=[100.] * d, power=2)
    matrix = self.evaluate(_operator_from_kernel(kernel, d).to_dense())

    tol = np.finfo(matrix.dtype).eps * np.prod(grid_shape) * 50
    self.assert_real_symmetric(matrix, tol)
    self.assertAllClose(np.ones_like(matrix), matrix, rtol=tol)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorCirculantBaseTest(object):
  # [CLASS DOC]
  # HELPER: shared mixin for the one-dimensional circulant operator test cases. Not a test case
  # itself (plain object base); supplies per-dtype absolute and relative tolerances, a device
  # context hook, the mapping from operator shape to spectrum shape, and an explicit
  # construction of the dense circulant matrix from a spectrum used as the reference value.
  """Common class for circulant tests."""

  _atol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 2e-7,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 2e-7,
  }
  _rtol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 2e-7,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 2e-7,
  }

  @contextlib.contextmanager
  def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
    # [FUNCTION DOC]
    # HELPER: device-context override used by the test framework (generator context manager).
    # KNOWN ISSUE: the docstring claims the FFT operation mapping is overwritten, but the code
    # only delegates unchanged to the base test case implementation and yields its session; no
    # FFT mapping is modified here.
    # Inputs: session, whether to use the GPU, whether to force the GPU. Enters the base class
    # device-constraining context and yields the session it provides.
    """We overwrite the FFT operation mapping for testing."""
    with test.TestCase._constrain_devices_and_set_default(
        self, sess, use_gpu, force_gpu) as sess:
      yield sess

  def _shape_to_spectrum_shape(self, shape):
    # [FUNCTION DOC]
    # HELPER: maps an operator shape to the spectrum shape that produces it. Takes the operator
    # shape (batch dims plus rows and columns) and returns the same shape without the last
    # dimension, since a spectrum of batch shape plus N gives an operator of batch plus N by N.
    # If spectrum.shape = batch_shape + [N],
    # this creates an operator of shape batch_shape + [N, N]
    return shape[:-1]

  def _spectrum_to_circulant_1d(self, spectrum, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: reference-matrix builder for the 1D circulant tests. Converts the spectrum to
    # complex, derives the domain dimension from the expected shape, and returns zeros of the
    # given shape if the dimension is empty. Otherwise, for each basis vector it applies FFT,
    # multiplies by the spectrum, applies inverse FFT, and stacks the results as columns, then
    # casts to the requested dtype. Returns the dense (batch) circulant matrix.
    """Creates a circulant matrix from a spectrum.

    Intentionally done in an explicit yet inefficient way.  This provides a
    cross check to the main code that uses fancy reshapes.

    Args:
      spectrum: Float or complex `Tensor`.
      shape:  Python list.  Desired shape of returned matrix.
      dtype:  Type to cast the returned matrix to.

    Returns:
      Circulant (batch) matrix of desired `dtype`.
    """
    spectrum = _to_complex(spectrum)
    spectrum_shape = self._shape_to_spectrum_shape(shape)
    domain_dimension = spectrum_shape[-1]
    if not domain_dimension:
      return array_ops.zeros(shape, dtype)

    # Explicitly compute the action of spectrum on basis vectors.
    matrix_rows = []
    for m in range(domain_dimension):
      x = np.zeros([domain_dimension])
      # x is a basis vector.
      x[m] = 1.0
      fft_x = fft_ops.fft(math_ops.cast(x, spectrum.dtype))
      h_convolve_x = fft_ops.ifft(spectrum * fft_x)
      matrix_rows.append(h_convolve_x)
    matrix = array_ops_stack.stack(matrix_rows, axis=-1)
    return math_ops.cast(matrix, dtype)


class LinearOperatorCirculantTestSelfAdjointOperator(
    LinearOperatorCirculantBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Derived-class test case for the 1D circulant operator with a real spectrum (so the operator is
  # self-adjoint, though its dtype is complex). Plugs into the shared operator test utility by
  # supplying dtypes, optional tests and an operator-plus-dense-matrix factory, and adds tests for
  # the imaginary part, tape safety and variable conversion.
  """Test of LinearOperatorCirculant when operator is self-adjoint.

  Real spectrum <==> Self adjoint operator.
  Note that when the spectrum is real, the operator may still be complex.
  """

  @staticmethod
  def dtypes_to_test():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Returns the dtypes to test: complex64 and complex128
    # only, because the matrix of a real-spectrum circulant operator is generally complex.
    # This operator will always be complex because, although the spectrum is
    # real, the matrix will not be real.
    return [dtypes.complex64, dtypes.complex128]

  @staticmethod
  def optional_tests():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Takes no inputs and returns the names of optional
    # generated tests to enable: matmul and solve with an input of the same type as the operator.
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
    ]

  def operator_and_matrix(self,
                          shape_info,
                          dtype,
                          use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Factory for the base test utility. Inputs: shape_info (record with the operator shape),
    # dtype, use_placeholder (wrap the spectrum in a placeholder with unknown shape) and
    # ensure_self_adjoint_and_pd (use the absolute value of the spectrum and flag the operator
    # positive definite). Builds a real random spectrum bounded away from zero (values from one
    # to two in magnitude, random sign), casts it to dtype, constructs the 1D circulant operator
    # flagged self-adjoint, and builds the reference dense matrix with the explicit helper.
    # Returns the pair (operator, dense matrix).
    shape = shape_info.shape
    # For this test class, we are creating real spectrums.
    # We also want the spectrum to have eigenvalues bounded away from zero.
    #
    # spectrum is bounded away from zero.
    spectrum = linear_operator_test_util.random_sign_uniform(
        shape=self._shape_to_spectrum_shape(shape), minval=1., maxval=2.)
    if ensure_self_adjoint_and_pd:
      spectrum = math_ops.abs(spectrum)
    # If dtype is complex, cast spectrum to complex.  The imaginary part will be
    # zero, so the operator will still be self-adjoint.
    spectrum = math_ops.cast(spectrum, dtype)

    lin_op_spectrum = spectrum

    if use_placeholder:
      lin_op_spectrum = array_ops.placeholder_with_default(spectrum, shape=None)

    operator = linalg.LinearOperatorCirculant(
        lin_op_spectrum,
        is_self_adjoint=True,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None,
        input_output_dtype=dtype)

    mat = self._spectrum_to_circulant_1d(spectrum, shape, dtype=dtype)

    return operator, mat

  @test_util.disable_xla("No registered Const")
  def test_simple_hermitian_spectrum_gives_operator_with_zero_imag_part(self):
    # [FUNCTION DOC]
    # Builds an operator from the small Hermitian spectrum (one, i, minus i) with complex64 dtype,
    # densifies it, and asserts the imaginary part is zero within a few machine epsilons of
    # float32. Skipped under XLA.
    with self.cached_session():
      spectrum = math_ops.cast([1. + 0j, 1j, -1j], dtypes.complex64)
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix = operator.to_dense()
      imag_matrix = math_ops.imag(matrix)
      eps = np.finfo(np.float32).eps
      np.testing.assert_allclose(
          0, self.evaluate(imag_matrix), rtol=0, atol=eps * 3)

  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Builds an operator whose spectrum is a variable (two ones) flagged self-adjoint and runs the
    # shared check that operator methods are differentiable with a gradient tape.
    spectrum = variables_module.Variable(
        math_ops.cast([1. + 0j, 1. + 0j], dtypes.complex64))
    operator = linalg.LinearOperatorCirculant(spectrum, is_self_adjoint=True)
    self.check_tape_safe(operator)

  def test_convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # Builds a self-adjoint operator over a variable spectrum, initializes the variable in a
    # cached session and runs the shared check that converting the operator's variables to
    # tensors works.
    spectrum = variables_module.Variable(
        math_ops.cast([1. + 0j, 1. + 0j], dtypes.complex64))
    operator = linalg.LinearOperatorCirculant(spectrum, is_self_adjoint=True)
    with self.cached_session() as sess:
      sess.run([spectrum.initializer])
      self.check_convert_variables_to_tensors(operator)


class LinearOperatorCirculantTestHermitianSpectrum(
    LinearOperatorCirculantBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Derived-class test case for the 1D circulant operator with a Hermitian spectrum (real valued
  # operator), tested with real and complex dtypes. Disables TF32 during tests, supplies the
  # operator-plus-dense-matrix factory built from a symmetric kernel spectrum, and adds a zero
  # imaginary part test and a tape-safety test.
  """Test of LinearOperatorCirculant when the spectrum is Hermitian.

  Hermitian spectrum <==> Real valued operator.  We test both real and complex
  dtypes here though.  So in some cases the matrix will be complex but with
  zero imaginary part.
  """

  def tearDown(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the base class tearDown, so parent cleanup is skipped.
    # Restores the TensorFloat-32 execution setting saved in setUp. Takes no inputs, returns
    # nothing; side effect is changing the global TF32 flag back to its original value.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the base class setUp, so parent fixture setup is skipped.
    # Test fixture setup. Saves the current TensorFloat-32 execution flag on the instance and then
    # disables TF32 so float32 results match the reference matrices closely. Takes no inputs and
    # returns nothing; side effect is changing the global TF32 flag.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)

  @staticmethod
  def optional_tests():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Takes no inputs and returns the names of optional
    # generated tests to enable: matmul and solve with an input of the same type as the operator.
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
    ]

  def operator_and_matrix(self,
                          shape_info,
                          dtype,
                          use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Factory for the base test utility. Inputs: shape_info (record with the operator shape),
    # dtype, use_placeholder (wrap the spectrum in a placeholder with unknown shape) and
    # ensure_self_adjoint_and_pd (use a positive spectrum and flag the operator self-adjoint and
    # positive definite). Builds a Hermitian spectrum from the symmetric exponential power kernel
    # for one dimension, constructs the 1D circulant operator, and builds the reference matrix
    # with the explicit helper. Returns the pair (operator, dense matrix).
    shape = shape_info.shape
    spectrum = _spectrum_for_symmetric_circulant(
        spectrum_shape=self._shape_to_spectrum_shape(shape),
        d=1,
        ensure_self_adjoint_and_pd=ensure_self_adjoint_and_pd,
        dtype=dtype)

    lin_op_spectrum = spectrum

    if use_placeholder:
      lin_op_spectrum = array_ops.placeholder_with_default(spectrum, shape=None)

    operator = linalg.LinearOperatorCirculant(
        lin_op_spectrum,
        input_output_dtype=dtype,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
    )

    mat = self._spectrum_to_circulant_1d(spectrum, shape, dtype=dtype)

    return operator, mat

  @test_util.disable_xla("No registered Const")
  def test_simple_hermitian_spectrum_gives_operator_with_zero_imag_part(self):
    # [FUNCTION DOC]
    # Builds an operator from the small Hermitian spectrum (one, i, minus i) with complex64 dtype,
    # densifies it, and asserts the imaginary part is zero within a few machine epsilons of
    # float32. Skipped under XLA.
    with self.cached_session():
      spectrum = math_ops.cast([1. + 0j, 1j, -1j], dtypes.complex64)
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix = operator.to_dense()
      imag_matrix = math_ops.imag(matrix)
      eps = np.finfo(np.float32).eps
      np.testing.assert_allclose(
          0, self.evaluate(imag_matrix), rtol=0, atol=eps * 3)

  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Builds an operator over a variable spectrum (one, one plus i) explicitly flagged not
    # self-adjoint and runs the shared check that operator methods are differentiable with a
    # gradient tape.
    spectrum = variables_module.Variable(
        math_ops.cast([1. + 0j, 1. + 1j], dtypes.complex64))
    operator = linalg.LinearOperatorCirculant(spectrum, is_self_adjoint=False)
    self.check_tape_safe(operator)


class LinearOperatorCirculantTestNonHermitianSpectrum(
    LinearOperatorCirculantBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Derived-class test case for the 1D circulant operator with a non-Hermitian (random complex)
  # spectrum, giving a complex valued operator tested with complex dtypes only. Skips cholesky and
  # eigenvalue tests, supplies the operator factory (also asserting the recorded constructor
  # parameters), and adds tests for adjoint and inverse types, singularity and positive-definite
  # assertions, kernel round trips, and validation of the self-adjoint hint for real spectra.
  """Test of LinearOperatorCirculant when the spectrum is not Hermitian.

  Non-Hermitian spectrum <==> Complex valued operator.
  We test only complex dtypes here.
  """

  @staticmethod
  def dtypes_to_test():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Returns the dtypes to test: complex64 and complex128
    # only, because the operator from a non-Hermitian spectrum is complex valued.
    return [dtypes.complex64, dtypes.complex128]

  # Skip Cholesky since we are explicitly testing non-hermitian
  # spectra.
  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Returns the generated test names to skip: cholesky and
    # eigenvalue tests, since non-Hermitian spectra give operators that are not self-adjoint.
    return ["cholesky", "eigvalsh"]

  @staticmethod
  def optional_tests():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Takes no inputs and returns the names of optional
    # generated tests to enable: matmul and solve with an input of the same type as the operator.
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
    ]

  def operator_and_matrix(self,
                          shape_info,
                          dtype,
                          use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Factory for the base test utility. Inputs: shape_info (record with the operator shape),
    # dtype, use_placeholder (wrap the spectrum in a placeholder with unknown shape) and
    # ensure_self_adjoint_and_pd (ignored; deleted since a random complex spectrum cannot be made
    # self-adjoint). Builds a random complex spectrum with magnitudes one to two, constructs the 1D
    # circulant operator, asserts the operator's recorded constructor parameters equal the
    # expected mapping (dtype, all flags unset except square true, default name, the spectrum),
    # and builds the reference matrix. Returns the pair (operator, dense matrix).
    del ensure_self_adjoint_and_pd
    shape = shape_info.shape
    # Will be well conditioned enough to get accurate solves.
    spectrum = linear_operator_test_util.random_sign_uniform(
        shape=self._shape_to_spectrum_shape(shape),
        dtype=dtype,
        minval=1.,
        maxval=2.)

    lin_op_spectrum = spectrum

    if use_placeholder:
      lin_op_spectrum = array_ops.placeholder_with_default(spectrum, shape=None)

    operator = linalg.LinearOperatorCirculant(
        lin_op_spectrum, input_output_dtype=dtype)

    self.assertEqual(
        operator.parameters,
        {
            "input_output_dtype": dtype,
            "is_non_singular": None,
            "is_positive_definite": None,
            "is_self_adjoint": None,
            "is_square": True,
            "name": "LinearOperatorCirculant",
            "spectrum": lin_op_spectrum,
        })

    mat = self._spectrum_to_circulant_1d(spectrum, shape, dtype=dtype)

    return operator, mat

  @test_util.disable_xla("No registered Const")
  def test_simple_hermitian_spectrum_gives_operator_with_zero_imag_part(self):
    # [FUNCTION DOC]
    # Builds an operator from the small Hermitian spectrum (one, i, minus i) with complex64 dtype,
    # densifies it, and asserts the imaginary part is zero within a few machine epsilons of
    # float32. Skipped under XLA.
    with self.cached_session():
      spectrum = math_ops.cast([1. + 0j, 1j, -1j], dtypes.complex64)
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix = operator.to_dense()
      imag_matrix = math_ops.imag(matrix)
      eps = np.finfo(np.float32).eps
      np.testing.assert_allclose(
          0, self.evaluate(imag_matrix), rtol=0, atol=eps * 3)

  def test_adjoint_output(self):
    # [FUNCTION DOC]
    # Takes the adjoint of a 1D circulant operator built from a small complex spectrum and
    # asserts the result is again a 1D circulant operator whose spectrum is complex.
    spectrum = math_ops.cast([1. + 0j, 1j, -1j], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(
        spectrum, input_output_dtype=dtypes.complex64)
    with self.cached_session():
      op_adjoint = operator.adjoint()
      self.assertIsInstance(op_adjoint, linalg.LinearOperatorCirculant)
      self.assertTrue(op_adjoint.spectrum.dtype.is_complex)

  def test_inverse_output(self):
    # [FUNCTION DOC]
    # Takes the inverse of a 1D circulant operator built from a small complex spectrum and asserts
    # the result is again a 1D circulant operator.
    spectrum = math_ops.cast([1. + 0j, 1j, -1j], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(
        spectrum, input_output_dtype=dtypes.complex64)
    with self.cached_session():
      op_inverse = operator.inverse()
      self.assertIsInstance(op_inverse, linalg.LinearOperatorCirculant)

  def test_simple_positive_real_spectrum_gives_self_adjoint_pos_def_oper(self):
    # [FUNCTION DOC]
    # Builds an operator from a positive real spectrum (six, four, two) in complex64, densifies it
    # together with its adjoint in one session run, asserts they are equal, and asserts that the
    # positive-definite and self-adjoint assertion ops evaluate without error.
    with self.cached_session() as sess:
      spectrum = math_ops.cast([6., 4, 2], dtypes.complex64)
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix, matrix_h = sess.run(
          [operator.to_dense(),
           linalg.adjoint(operator.to_dense())])
      self.assertAllClose(matrix, matrix_h)
      self.evaluate(operator.assert_positive_definite())  # Should not fail
      self.evaluate(operator.assert_self_adjoint())  # Should not fail

  def test_defining_operator_using_real_convolution_kernel(self):
    # [FUNCTION DOC]
    # Takes the FFT of the real kernel (one, two, one) as the spectrum, builds the 1D operator,
    # asserts its dtype is complex64 and that the dense matrix has zero imaginary part (the
    # spectrum is Hermitian so the operator is real).
    with self.cached_session():
      convolution_kernel = [1., 2., 1.]
      spectrum = fft_ops.fft(
          math_ops.cast(convolution_kernel, dtypes.complex64))

      # spectrum is shape [3] ==> operator is shape [3, 3]
      # spectrum is Hermitian ==> operator is real.
      operator = linalg.LinearOperatorCirculant(spectrum)

      # Allow for complex output so we can make sure it has zero imag part.
      self.assertEqual(operator.dtype, dtypes.complex64)

      matrix = self.evaluate(operator.to_dense())
      np.testing.assert_allclose(0, np.imag(matrix), atol=1e-6)

  @test_util.run_v1_only("currently failing on v2")
  def test_hermitian_spectrum_gives_operator_with_zero_imag_part(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: decorated as graph-mode (v1) only because it "currently fails on v2"; the test
    # is therefore not exercised under TF2 behavior.
    # Builds a spectrum as the FFT of a random real kernel of shape three by four (so it is
    # Hermitian), builds the operator with complex64 dtype, densifies it and asserts the imaginary
    # part is zero within a small multiple of float32 epsilon.
    with self.cached_session():
      # Make spectrum the FFT of a real convolution kernel h.  This ensures that
      # spectrum is Hermitian.
      h = linear_operator_test_util.random_normal(shape=(3, 4))
      spectrum = fft_ops.fft(math_ops.cast(h, dtypes.complex64))
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix = operator.to_dense()
      imag_matrix = math_ops.imag(matrix)
      eps = np.finfo(np.float32).eps
      np.testing.assert_allclose(
          0, self.evaluate(imag_matrix), rtol=0, atol=eps * 3 * 4)

  def test_convolution_kernel_same_as_first_row_of_to_dense(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says first row but the assertion compares against the first column
    # (last index fixed at zero); for a circulant matrix generated this way those coincide only
    # for symmetric kernels.
    # Builds an operator from a batch of two real spectra of length three, asks for its
    # convolution kernel and dense matrix, asserts their shapes (two by three, two by three by
    # three) and that the kernel equals the first column of each dense matrix.
    spectrum = [[3., 2., 1.], [2., 1.5, 1.]]
    with self.cached_session():
      operator = linalg.LinearOperatorCirculant(spectrum)
      h = operator.convolution_kernel()
      c = operator.to_dense()

      self.assertAllEqual((2, 3), h.shape)
      self.assertAllEqual((2, 3, 3), c.shape)
      self.assertAllClose(self.evaluate(h), self.evaluate(c)[:, :, 0])

  def test_assert_non_singular_fails_for_singular_operator(self):
    # [FUNCTION DOC]
    # Builds an operator whose spectrum contains a zero and asserts that evaluating the
    # non-singular assertion op fails with an op error mentioning a singular operator.
    spectrum = math_ops.cast([0 + 0j, 4 + 0j, 2j + 2], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(spectrum)
    with self.cached_session():
      with self.assertRaisesOpError("Singular operator"):
        self.evaluate(operator.assert_non_singular())

  def test_assert_non_singular_does_not_fail_for_non_singular_operator(self):
    # [FUNCTION DOC]
    # Builds an operator whose spectrum has no zero entries and asserts the non-singular assertion
    # op evaluates without error.
    spectrum = math_ops.cast([-3j, 4 + 0j, 2j + 2], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(spectrum)
    with self.cached_session():
      self.evaluate(operator.assert_non_singular())  # Should not fail

  def test_assert_positive_definite_fails_for_non_positive_definite(self):
    # [FUNCTION DOC]
    # Builds an operator whose spectrum has an entry with zero real part and asserts the
    # positive-definite assertion op fails with an op error saying it is not positive definite.
    spectrum = math_ops.cast([6. + 0j, 4 + 0j, 2j], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(spectrum)
    with self.cached_session():
      with self.assertRaisesOpError("Not positive definite"):
        self.evaluate(operator.assert_positive_definite())

  def test_assert_positive_definite_does_not_fail_when_pos_def(self):
    # [FUNCTION DOC]
    # Builds an operator whose spectrum entries all have positive real part and asserts the
    # positive-definite assertion op evaluates without error.
    spectrum = math_ops.cast([6. + 0j, 4 + 0j, 2j + 2], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(spectrum)
    with self.cached_session():
      self.evaluate(operator.assert_positive_definite())  # Should not fail

  def test_real_spectrum_and_not_self_adjoint_hint_raises(self):
    # [FUNCTION DOC]
    # Constructs a 1D operator from a real spectrum while explicitly claiming it is not
    # self-adjoint and asserts a value error stating a real spectrum is always self-adjoint.
    spectrum = [1., 2.]
    with self.assertRaisesRegex(ValueError, "real.*always.*self-adjoint"):
      linalg.LinearOperatorCirculant(spectrum, is_self_adjoint=False)

  def test_real_spectrum_auto_sets_is_self_adjoint_to_true(self):
    # [FUNCTION DOC]
    # Constructs a 1D operator from a real spectrum without any self-adjoint hint and asserts the
    # operator reports itself as self-adjoint.
    spectrum = [1., 2.]
    operator = linalg.LinearOperatorCirculant(spectrum)
    self.assertTrue(operator.is_self_adjoint)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorCirculant2DBaseTest(object):
  # [CLASS DOC]
  # HELPER: shared mixin for the two-dimensional (block) circulant operator test cases. Not a test
  # case itself; supplies per-dtype tolerances, a device context hook, the operator shapes to
  # test, optional test names, the mapping from operator shape to a two-dimensional spectrum shape
  # and an explicit loop-based construction of the dense block circulant matrix from a spectrum.
  """Common class for 2D circulant tests."""

  _atol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 2e-7,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 2e-7,
  }
  _rtol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 2e-7,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 2e-7,
  }

  @contextlib.contextmanager
  def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
    # [FUNCTION DOC]
    # HELPER: device-context override used by the test framework (generator context manager).
    # KNOWN ISSUE: the docstring claims the FFT operation mapping is overwritten, but the code
    # only delegates unchanged to the base test case implementation and yields its session; no
    # FFT mapping is modified here.
    # Inputs: session, whether to use the GPU, whether to force the GPU. Enters the base class
    # device-constraining context and yields the session it provides.
    """We overwrite the FFT operation mapping for testing."""
    with test.TestCase._constrain_devices_and_set_default(
        self, sess, use_gpu, force_gpu) as sess:
      yield sess

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Returns shape-info records for the operator shapes
    # to test: empty, 1x1, and batched shapes (1 by 6x6, 3 by 4x4, 2 by 1 by 3x3).
    shape_info = linear_operator_test_util.OperatorShapesInfo
    # non-batch operators (n, n) and batch operators.
    return [
        shape_info((0, 0)),
        shape_info((1, 1)),
        shape_info((1, 6, 6)),
        shape_info((3, 4, 4)),
        shape_info((2, 1, 3, 3))
    ]

  @staticmethod
  def optional_tests():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Takes no inputs and returns the names of optional
    # generated tests to enable: matmul and solve with an input of the same type as the operator.
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
    ]

  def _shape_to_spectrum_shape(self, shape):
    # [FUNCTION DOC]
    # HELPER: maps an operator shape to a two-dimensional spectrum shape that produces it, using
    # a fixed lookup of the five supported operator shapes (for example 1 by 6x6 maps to 1 by 2 by
    # 3). Input is the operator shape as a tuple; returns the spectrum shape tuple; raises a value
    # error for any unsupported shape.
    """Get a spectrum shape that will make an operator of desired shape."""
    # This 2D block circulant operator takes a spectrum of shape
    # batch_shape + [N0, N1],
    # and creates and operator of shape
    # batch_shape + [N0*N1, N0*N1]
    if shape == (0, 0):
      return (0, 0)
    elif shape == (1, 1):
      return (1, 1)
    elif shape == (1, 6, 6):
      return (1, 2, 3)
    elif shape == (3, 4, 4):
      return (3, 2, 2)
    elif shape == (2, 1, 3, 3):
      return (2, 1, 3, 1)
    else:
      raise ValueError("Unhandled shape: %s" % shape)

  def _spectrum_to_circulant_2d(self, spectrum, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: reference-matrix builder for the 2D circulant tests. Converts the spectrum to
    # complex, derives the block grid from the spectrum shape, and returns zeros of the given
    # shape if the domain is empty. Otherwise, for every grid-position basis vector it applies a
    # 2D FFT, multiplies by the spectrum, applies the inverse 2D FFT, flattens the result to a
    # vector per batch member, and stacks the results as columns; finally casts to the requested
    # dtype. Returns the dense block-circulant (batch) matrix.
    """Creates a block circulant matrix from a spectrum.

    Intentionally done in an explicit yet inefficient way.  This provides a
    cross check to the main code that uses fancy reshapes.

    Args:
      spectrum: Float or complex `Tensor`.
      shape:  Python list.  Desired shape of returned matrix.
      dtype:  Type to cast the returned matrix to.

    Returns:
      Block circulant (batch) matrix of desired `dtype`.
    """
    spectrum = _to_complex(spectrum)
    spectrum_shape = self._shape_to_spectrum_shape(shape)
    domain_dimension = spectrum_shape[-1]
    if not domain_dimension:
      return array_ops.zeros(shape, dtype)

    block_shape = spectrum_shape[-2:]

    # Explicitly compute the action of spectrum on basis vectors.
    matrix_rows = []
    for n0 in range(block_shape[0]):
      for n1 in range(block_shape[1]):
        x = np.zeros(block_shape)
        # x is a basis vector.
        x[n0, n1] = 1.0
        fft_x = fft_ops.fft2d(math_ops.cast(x, spectrum.dtype))
        h_convolve_x = fft_ops.ifft2d(spectrum * fft_x)
        # We want the flat version of the action of the operator on a basis
        # vector, not the block version.
        h_convolve_x = array_ops.reshape(h_convolve_x, shape[:-1])
        matrix_rows.append(h_convolve_x)
    matrix = array_ops_stack.stack(matrix_rows, axis=-1)
    return math_ops.cast(matrix, dtype)


class LinearOperatorCirculant2DTestHermitianSpectrum(
    LinearOperatorCirculant2DBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Derived-class test case for the 2D block-circulant operator with a Hermitian spectrum (real
  # valued operator), tested with real and complex dtypes. Disables TF32 during tests and supplies
  # the operator-plus-dense-matrix factory built from a symmetric two-dimensional kernel spectrum.
  """Test of LinearOperatorCirculant2D when the spectrum is Hermitian.

  Hermitian spectrum <==> Real valued operator.  We test both real and complex
  dtypes here though.  So in some cases the matrix will be complex but with
  zero imaginary part.
  """

  def tearDown(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the base class tearDown, so parent cleanup is skipped.
    # Restores the TensorFloat-32 execution setting saved in setUp. Takes no inputs, returns
    # nothing; side effect is changing the global TF32 flag back to its original value.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the base class setUp, so parent fixture setup is skipped.
    # Test fixture setup. Saves the current TensorFloat-32 execution flag on the instance and then
    # disables TF32 so float32 results match the reference matrices closely. Takes no inputs and
    # returns nothing; side effect is changing the global TF32 flag.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)

  def operator_and_matrix(self,
                          shape_info,
                          dtype,
                          use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Factory for the base test utility. Inputs: shape_info (record with the operator shape),
    # dtype, use_placeholder (wrap the spectrum in a placeholder with unknown shape) and
    # ensure_self_adjoint_and_pd (use a positive spectrum and flag the operator self-adjoint and
    # positive definite). Builds a Hermitian spectrum from the symmetric kernel for two
    # dimensions, constructs the 2D circulant operator, asserts its recorded constructor
    # parameters match the expected mapping, and builds the reference matrix. Returns the pair
    # (operator, dense matrix).
    shape = shape_info.shape
    spectrum = _spectrum_for_symmetric_circulant(
        spectrum_shape=self._shape_to_spectrum_shape(shape),
        d=2,
        ensure_self_adjoint_and_pd=ensure_self_adjoint_and_pd,
        dtype=dtype)

    lin_op_spectrum = spectrum

    if use_placeholder:
      lin_op_spectrum = array_ops.placeholder_with_default(spectrum, shape=None)

    operator = linalg.LinearOperatorCirculant2D(
        lin_op_spectrum,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        input_output_dtype=dtype)

    self.assertEqual(
        operator.parameters,
        {
            "input_output_dtype": dtype,
            "is_non_singular": None,
            "is_positive_definite": (
                True if ensure_self_adjoint_and_pd else None),
            "is_self_adjoint": (
                True if ensure_self_adjoint_and_pd else None),
            "is_square": True,
            "name": "LinearOperatorCirculant2D",
            "spectrum": lin_op_spectrum,
        })

    mat = self._spectrum_to_circulant_2d(spectrum, shape, dtype=dtype)

    return operator, mat


class LinearOperatorCirculant2DTestNonHermitianSpectrum(
    LinearOperatorCirculant2DBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Derived-class test case for the 2D block-circulant operator with a non-Hermitian (random
  # complex) spectrum, tested with complex dtypes only, skipping cholesky and eigenvalue tests.
  # Supplies the operator factory and adds tests for real-spectrum behavior, adjoint and inverse
  # types, singularity and positive-definite assertions, self-adjoint hint validation, invalid
  # spectrum rank and tape safety.
  """Test of LinearOperatorCirculant when the spectrum is not Hermitian.

  Non-Hermitian spectrum <==> Complex valued operator.
  We test only complex dtypes here.
  """

  @staticmethod
  def dtypes_to_test():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Returns the dtypes to test: complex64 and complex128
    # only, because the operator from a non-Hermitian spectrum is complex valued.
    return [dtypes.complex64, dtypes.complex128]

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Returns the generated test names to skip: cholesky and
    # eigenvalue tests, since non-Hermitian spectra give operators that are not self-adjoint.
    return ["cholesky", "eigvalsh"]

  def operator_and_matrix(self,
                          shape_info,
                          dtype,
                          use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Factory for the base test utility. Inputs: shape_info (record with the operator shape),
    # dtype, use_placeholder (wrap the spectrum in a placeholder with unknown shape) and
    # ensure_self_adjoint_and_pd (ignored; deleted). Builds a random complex spectrum with
    # magnitudes one to two, constructs the 2D circulant operator, asserts its recorded
    # constructor parameters match the expected mapping, and builds the reference matrix. Returns
    # the pair (operator, dense matrix).
    del ensure_self_adjoint_and_pd
    shape = shape_info.shape
    # Will be well conditioned enough to get accurate solves.
    spectrum = linear_operator_test_util.random_sign_uniform(
        shape=self._shape_to_spectrum_shape(shape),
        dtype=dtype,
        minval=1.,
        maxval=2.)

    lin_op_spectrum = spectrum

    if use_placeholder:
      lin_op_spectrum = array_ops.placeholder_with_default(spectrum, shape=None)

    operator = linalg.LinearOperatorCirculant2D(
        lin_op_spectrum, input_output_dtype=dtype)

    self.assertEqual(
        operator.parameters,
        {
            "input_output_dtype": dtype,
            "is_non_singular": None,
            "is_positive_definite": None,
            "is_self_adjoint": None,
            "is_square": True,
            "name": "LinearOperatorCirculant2D",
            "spectrum": lin_op_spectrum,
        }
    )

    mat = self._spectrum_to_circulant_2d(spectrum, shape, dtype=dtype)

    return operator, mat

  def test_real_hermitian_spectrum_gives_real_symmetric_operator(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: lives in the 2D test class but constructs the one-dimensional circulant
    # operator from a three by three spectrum (treated as a batch of three length-three
    # spectra), so it does not exercise the 2D operator.
    # Builds an operator from a real symmetric spectrum, densifies it along with its transpose and
    # imaginary part, asserts dtype complex64, zero imaginary part and that the matrix equals its
    # transpose.
    with self.cached_session():  # Necessary for fft_kernel_label_map
      # This is a real and hermitian spectrum.
      spectrum = [[1., 2., 2.], [3., 4., 4.], [3., 4., 4.]]
      operator = linalg.LinearOperatorCirculant(spectrum)

      matrix_tensor = operator.to_dense()
      self.assertEqual(matrix_tensor.dtype, dtypes.complex64)
      matrix_t = array_ops.matrix_transpose(matrix_tensor)
      imag_matrix = math_ops.imag(matrix_tensor)
      matrix, matrix_transpose, imag_matrix = self.evaluate(
          [matrix_tensor, matrix_t, imag_matrix])

      np.testing.assert_allclose(0, imag_matrix, atol=1e-6)
      self.assertAllClose(matrix, matrix_transpose, atol=1e-6)

  def test_real_spectrum_gives_self_adjoint_operator(self):
    # [FUNCTION DOC]
    # Builds a 2D operator from a random real three by three spectrum, densifies it and its
    # adjoint, asserts dtype complex64 and that the matrix equals its adjoint within tolerance.
    with self.cached_session():
      # This is a real and hermitian spectrum.
      spectrum = linear_operator_test_util.random_normal(
          shape=(3, 3), dtype=dtypes.float32)
      operator = linalg.LinearOperatorCirculant2D(spectrum)

      matrix_tensor = operator.to_dense()
      self.assertEqual(matrix_tensor.dtype, dtypes.complex64)
      matrix_h = linalg.adjoint(matrix_tensor)
      matrix, matrix_h = self.evaluate([matrix_tensor, matrix_h])
      self.assertAllClose(matrix, matrix_h, atol=1e-5)

  def test_adjoint_output(self):
    # [FUNCTION DOC]
    # Takes the adjoint of a 2D circulant operator built from a small complex two by two spectrum
    # and asserts the result is again a 2D circulant operator with a complex spectrum.
    spectrum = math_ops.cast([[-3j, 4 + 0j], [2j + 2, 3. + 0j]],
                             dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.cached_session():
      op_adjoint = operator.adjoint()
      self.assertIsInstance(op_adjoint, linalg.LinearOperatorCirculant2D)
      self.assertTrue(op_adjoint.spectrum.dtype.is_complex)

  def test_inverse_output(self):
    # [FUNCTION DOC]
    # Takes the inverse of a 2D circulant operator built from a small complex two by two spectrum
    # and asserts the result is again a 2D circulant operator.
    spectrum = math_ops.cast([[-3j, 4 + 0j], [2j + 2, 3. + 0j]],
                             dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.cached_session():
      op_inverse = operator.inverse()
      self.assertIsInstance(op_inverse, linalg.LinearOperatorCirculant2D)

  def test_assert_non_singular_fails_for_singular_operator(self):
    # [FUNCTION DOC]
    # Builds a 2D operator whose spectrum contains a zero entry and asserts that evaluating the
    # non-singular assertion op fails with an op error mentioning a singular operator.
    spectrum = math_ops.cast([[0 + 0j, 4 + 0j], [2j + 2, 3. + 0j]],
                             dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.cached_session():
      with self.assertRaisesOpError("Singular operator"):
        self.evaluate(operator.assert_non_singular())

  def test_assert_non_singular_does_not_fail_for_non_singular_operator(self):
    # [FUNCTION DOC]
    # Builds a 2D operator whose spectrum has no zero entries and asserts the non-singular
    # assertion op evaluates without error.
    spectrum = math_ops.cast([[-3j, 4 + 0j], [2j + 2, 3. + 0j]],
                             dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.cached_session():
      self.evaluate(operator.assert_non_singular())  # Should not fail

  def test_assert_positive_definite_fails_for_non_positive_definite(self):
    # [FUNCTION DOC]
    # Builds a 2D operator whose spectrum has an entry with zero real part and asserts the
    # positive-definite assertion op fails with an op error saying it is not positive definite.
    spectrum = math_ops.cast([[6. + 0j, 4 + 0j], [2j, 3. + 0j]],
                             dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.cached_session():
      with self.assertRaisesOpError("Not positive definite"):
        self.evaluate(operator.assert_positive_definite())

  def test_assert_positive_definite_does_not_fail_when_pos_def(self):
    # [FUNCTION DOC]
    # Builds a 2D operator whose spectrum entries all have positive real part and asserts the
    # positive-definite assertion op evaluates without error.
    spectrum = math_ops.cast([[6. + 0j, 4 + 0j], [2j + 2, 3. + 0j]],
                             dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.cached_session():
      self.evaluate(operator.assert_positive_definite())  # Should not fail

  def test_real_spectrum_and_not_self_adjoint_hint_raises(self):
    # [FUNCTION DOC]
    # Constructs a 2D operator from a real spectrum while explicitly claiming it is not
    # self-adjoint and asserts a value error stating a real spectrum is always self-adjoint.
    spectrum = [[1., 2.], [3., 4]]
    with self.assertRaisesRegex(ValueError, "real.*always.*self-adjoint"):
      linalg.LinearOperatorCirculant2D(spectrum, is_self_adjoint=False)

  def test_real_spectrum_auto_sets_is_self_adjoint_to_true(self):
    # [FUNCTION DOC]
    # Constructs a 2D operator from a real spectrum without any self-adjoint hint and asserts the
    # operator reports itself as self-adjoint.
    spectrum = [[1., 2.], [3., 4]]
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    self.assertTrue(operator.is_self_adjoint)

  def test_invalid_rank_raises(self):
    # [FUNCTION DOC]
    # Constructs a 2D operator from a one-dimensional spectrum and asserts a value error saying
    # the spectrum must have at least two dimensions.
    spectrum = array_ops.constant(np.float32(rng.rand(2)))
    with self.assertRaisesRegex(ValueError, "must have at least 2 dimensions"):
      linalg.LinearOperatorCirculant2D(spectrum)

  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Builds a 2D operator over a variable two by two complex spectrum and runs the shared check
    # that operator methods are differentiable with a gradient tape.
    spectrum = variables_module.Variable(
        math_ops.cast([[1. + 0j, 1. + 0j], [1. + 1j, 2. + 2j]],
                      dtypes.complex64))
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    self.check_tape_safe(operator)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorCirculant3DTest(test.TestCase):
  # [CLASS DOC]
  # Direct (non derived-class) tests of the 3D block-circulant operator: real spectra give
  # self-adjoint operators, kernels from real data give real operators, adjoint and inverse
  # types, and construction of a positive-definite operator from the Hermitian part of a positive
  # kernel. Holds per-dtype tolerances and a device context hook. Runs in graph and eager modes.
  """Simple test of the 3D case.  See also the 1D and 2D tests."""

  _atol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-7,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-7
  }
  _rtol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-7,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-7
  }

  @contextlib.contextmanager
  def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
    # [FUNCTION DOC]
    # HELPER: device-context override used by the test framework (generator context manager).
    # KNOWN ISSUE: the docstring claims the FFT operation mapping is overwritten, but the code
    # only delegates unchanged to the base test case implementation and yields its session; no
    # FFT mapping is modified here.
    # Inputs: session, whether to use the GPU, whether to force the GPU. Enters the base class
    # device-constraining context and yields the session it provides.
    """We overwrite the FFT operation mapping for testing."""
    with test.TestCase._constrain_devices_and_set_default(
        self, sess, use_gpu, force_gpu) as sess:
      yield sess

  def test_real_spectrum_gives_self_adjoint_operator(self):
    # [FUNCTION DOC]
    # Builds a 3D operator from a random real spectrum of shape two, two, three, five, asserts the
    # operator shape (batch of two, 30 by 30) and the recorded constructor parameters, densifies
    # the operator and its adjoint, and asserts dtype complex64, the dense shape and that the
    # matrix equals its adjoint.
    with self.cached_session():
      # This is a real and hermitian spectrum.
      spectrum = linear_operator_test_util.random_normal(
          shape=(2, 2, 3, 5), dtype=dtypes.float32)
      operator = linalg.LinearOperatorCirculant3D(spectrum)
      self.assertAllEqual((2, 2 * 3 * 5, 2 * 3 * 5), operator.shape)

      self.assertEqual(
          operator.parameters,
          {
              "input_output_dtype": dtypes.complex64,
              "is_non_singular": None,
              "is_positive_definite": None,
              "is_self_adjoint": None,
              "is_square": True,
              "name": "LinearOperatorCirculant3D",
              "spectrum": spectrum,
          })

      matrix_tensor = operator.to_dense()
      self.assertEqual(matrix_tensor.dtype, dtypes.complex64)
      matrix_h = linalg.adjoint(matrix_tensor)

      matrix, matrix_h = self.evaluate([matrix_tensor, matrix_h])
      self.assertAllEqual((2, 2 * 3 * 5, 2 * 3 * 5), matrix.shape)
      self.assertAllClose(matrix, matrix_h)

  def test_defining_operator_using_real_convolution_kernel(self):
    # [FUNCTION DOC]
    # Takes the 3D FFT of a random real kernel of shape two, two, three, five as the (Hermitian)
    # spectrum, builds the 3D operator, asserts its shape and complex64 dtype, and that the dense
    # matrix has the expected shape and zero imaginary part.
    with self.cached_session():
      convolution_kernel = linear_operator_test_util.random_normal(
          shape=(2, 2, 3, 5), dtype=dtypes.float32)
      # Convolution kernel is real ==> spectrum is Hermitian.
      spectrum = fft_ops.fft3d(
          math_ops.cast(convolution_kernel, dtypes.complex64))

      # spectrum is Hermitian ==> operator is real.
      operator = linalg.LinearOperatorCirculant3D(spectrum)
      self.assertAllEqual((2, 2 * 3 * 5, 2 * 3 * 5), operator.shape)

      # Allow for complex output so we can make sure it has zero imag part.
      self.assertEqual(operator.dtype, dtypes.complex64)
      matrix = self.evaluate(operator.to_dense())
      self.assertAllEqual((2, 2 * 3 * 5, 2 * 3 * 5), matrix.shape)
      np.testing.assert_allclose(0, np.imag(matrix), atol=1e-5)

  def test_adjoint_output(self):
    # [FUNCTION DOC]
    # Takes the adjoint of a 3D operator built from a random real spectrum and asserts the result
    # is again a 3D circulant operator.
    with self.cached_session():
      spectrum = linear_operator_test_util.random_normal(
          shape=(2, 2, 3, 5), dtype=dtypes.float32)
      operator = linalg.LinearOperatorCirculant3D(spectrum)
      op_adjoint = operator.adjoint()
      self.assertIsInstance(op_adjoint, linalg.LinearOperatorCirculant3D)

  def test_inverse_output(self):
    # [FUNCTION DOC]
    # Takes the inverse of a 3D operator built from a random real spectrum and asserts the result
    # is again a 3D circulant operator.
    spectrum = linear_operator_test_util.random_normal(
        shape=(2, 2, 3, 5), dtype=dtypes.float32)
    operator = linalg.LinearOperatorCirculant3D(spectrum)
    with self.cached_session():
      op_inverse = operator.inverse()
      self.assertIsInstance(op_inverse, linalg.LinearOperatorCirculant3D)

  def test_defining_spd_operator_by_taking_real_part(self):
    # [FUNCTION DOC]
    # Starts from a random positive real array, takes its inverse 3D FFT and keeps only the real
    # part (which isolates the Hermitian part of the kernel), re-applies the FFT to get a spectrum
    # that is Hermitian, real and positive, and builds the 3D operator. Asserts dtype complex64,
    # that the positive-definite assertion passes, zero imaginary part and symmetry of the dense
    # matrix. Then builds a second operator from the imaginary part (the anti-Hermitian
    # component) and asserts that its dense matrix equals its conjugate transpose and has zero
    # real part (an imaginary self-adjoint operator).
    with self.cached_session():  # Necessary for fft_kernel_label_map
      # S is real and positive.
      s = linear_operator_test_util.random_uniform(
          shape=(10, 2, 3, 4), dtype=dtypes.float32, minval=1., maxval=2.)

      # Let S = S1 + S2, the Hermitian and anti-hermitian parts.
      # S1 = 0.5 * (S + S^H), S2 = 0.5 * (S - S^H),
      # where ^H is the Hermitian transpose of the function:
      #    f(n0, n1, n2)^H := ComplexConjugate[f(N0-n0, N1-n1, N2-n2)].
      # We want to isolate S1, since
      #   S1 is Hermitian by construction
      #   S1 is real since S is
      #   S1 is positive since it is the sum of two positive kernels

      # IDFT[S] = IDFT[S1] + IDFT[S2]
      #         =      H1  +      H2
      # where H1 is real since it is Hermitian,
      # and H2 is imaginary since it is anti-Hermitian.
      ifft_s = fft_ops.ifft3d(math_ops.cast(s, dtypes.complex64))

      # Throw away H2, keep H1.
      real_ifft_s = math_ops.real(ifft_s)

      # This is the perfect spectrum!
      # spectrum = DFT[H1]
      #          = S1,
      fft_real_ifft_s = fft_ops.fft3d(
          math_ops.cast(real_ifft_s, dtypes.complex64))

      # S1 is Hermitian ==> operator is real.
      # S1 is real ==> operator is self-adjoint.
      # S1 is positive ==> operator is positive-definite.
      operator = linalg.LinearOperatorCirculant3D(fft_real_ifft_s)

      # Allow for complex output so we can check operator has zero imag part.
      self.assertEqual(operator.dtype, dtypes.complex64)
      matrix, matrix_t = self.evaluate([
          operator.to_dense(),
          array_ops.matrix_transpose(operator.to_dense())
      ])
      self.evaluate(operator.assert_positive_definite())  # Should not fail.
      np.testing.assert_allclose(0, np.imag(matrix), atol=1e-6)
      self.assertAllClose(matrix, matrix_t)

      # Just to test the theory, get S2 as well.
      # This should create an imaginary operator.
      # S2 is anti-Hermitian ==> operator is imaginary.
      # S2 is real ==> operator is self-adjoint.
      imag_ifft_s = math_ops.imag(ifft_s)
      fft_imag_ifft_s = fft_ops.fft3d(
          1j * math_ops.cast(imag_ifft_s, dtypes.complex64))
      operator_imag = linalg.LinearOperatorCirculant3D(fft_imag_ifft_s)

      matrix, matrix_h = self.evaluate([
          operator_imag.to_dense(),
          array_ops.matrix_transpose(math_ops.conj(operator_imag.to_dense()))
      ])
      self.assertAllClose(matrix, matrix_h)
      np.testing.assert_allclose(0, np.real(matrix), atol=1e-7)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(
      LinearOperatorCirculantTestSelfAdjointOperator)
  linear_operator_test_util.add_tests(
      LinearOperatorCirculantTestHermitianSpectrum)
  linear_operator_test_util.add_tests(
      LinearOperatorCirculantTestNonHermitianSpectrum)
  linear_operator_test_util.add_tests(
      LinearOperatorCirculant2DTestHermitianSpectrum)
  linear_operator_test_util.add_tests(
      LinearOperatorCirculant2DTestNonHermitianSpectrum)
  test.main()
