# [MODULE DOC]
# Tests for the signal FFT ops: complex FFT/IFFT of rank 1 to 3, real RFFT/IRFFT of rank 1 to 3,
# the N-dimensional variants (fftnd, ifftnd, rfftnd, irfftnd, GPU only), and the fftshift and
# ifftshift helpers. A shared base class compares TensorFlow results against NumPy (forward and
# backward, optionally through placeholders) and provides numerical gradient checks; two
# subclasses supply the complex and real transform specifics. Tests run in graph and eager modes.
# Import-time side effects: none (only defines the valid rank constant and test classes).
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
"""Tests for fft operations."""

import itertools
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.platform import test

VALID_FFT_RANKS = (1, 2, 3)


# TODO(rjryan): Investigate precision issues. We should be able to achieve
# better tolerances, at least for the complex128 tests.
class BaseFFTOpsTest(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base class for the complex and real FFT test classes.
  # Holds comparison and gradient-check helpers. Subclasses must provide the TensorFlow and NumPy
  # transform callables (_tf_fft, _tf_ifft, _np_fft, _np_ifft, the n-dimensional counterparts,
  # and the rank-to-op lookups); this class has no tests of its own.

  def _Compare_fftn(
      self,
      x,
      fft_length=None,
      axes=None,
      norm=None,
      use_placeholder=False,
      rtol=1e-4,
      atol=1e-4,
  ):
    # [FUNCTION DOC]
    # HELPER: runs both the forward and backward n-dimensional FFT comparisons.
    # Inputs: x (NumPy array), fft_length and axes (optional sequences), norm (optional
    # normalization mode), use_placeholder (feed through a placeholder in graph mode), rtol/atol
    # tolerances. Delegates to the forward and backward comparison helpers.
    self._CompareForward_fftn(
        x, fft_length, axes, norm, use_placeholder, rtol, atol
    )
    self._CompareBackward_fftn(
        x, fft_length, axes, norm, use_placeholder, rtol, atol
    )

  def _CompareForward_fftn(
      self,
      x,
      fft_length=None,
      axes=None,
      norm=None,
      use_placeholder=False,
      rtol=1e-4,
      atol=1e-4,
  ):
    # [FUNCTION DOC]
    # HELPER: compares the TensorFlow forward n-dimensional transform with the NumPy one.
    # Inputs as in the combined comparison helper. Computes the NumPy result, computes the
    # TensorFlow result (feeding x through a placeholder when requested) and asserts they are
    # close within the tolerances. Relies on subclass-provided transform callables.
    x_np = self._np_fftn(x, fft_length, axes, norm)
    if use_placeholder:
      x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
      x_tf = self._tf_fftn(x_ph, fft_length, axes, norm, feed_dict={x_ph: x})
    else:
      x_tf = self._tf_fftn(x, fft_length, axes, norm)
    self.assertAllClose(x_np, x_tf, rtol=rtol, atol=atol)

  def _CompareBackward_fftn(
      self,
      x,
      fft_length=None,
      axes=None,
      norm=None,
      use_placeholder=False,
      rtol=1e-4,
      atol=1e-4,
  ):
    # [FUNCTION DOC]
    # HELPER: compares the TensorFlow inverse n-dimensional transform with the NumPy one.
    # Same inputs and behavior as the forward comparison helper but using the inverse
    # transforms.
    x_np = self._np_ifftn(x, fft_length, axes, norm)
    if use_placeholder:
      x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
      x_tf = self._tf_ifftn(x_ph, fft_length, axes, norm, feed_dict={x_ph: x})
    else:
      x_tf = self._tf_ifftn(x, fft_length, axes, norm)
    self.assertAllClose(x_np, x_tf, rtol=rtol, atol=atol)

  def _compare(self, x, rank, fft_length=None, use_placeholder=False,
               rtol=1e-4, atol=1e-4):
    # [FUNCTION DOC]
    # HELPER: runs both the forward and backward fixed-rank FFT comparisons.
    # Inputs: x (NumPy array), rank (1, 2 or 3), fft_length (optional), use_placeholder,
    # rtol/atol tolerances. Delegates to the forward and backward comparison helpers.
    self._compare_forward(x, rank, fft_length, use_placeholder, rtol, atol)
    self._compare_backward(x, rank, fft_length, use_placeholder, rtol, atol)

  def _compare_forward(self, x, rank, fft_length=None, use_placeholder=False,
                       rtol=1e-4, atol=1e-4):
    # [FUNCTION DOC]
    # HELPER: compares the TensorFlow forward FFT of the given rank with the NumPy result.
    # Inputs: x, rank, fft_length (optional), use_placeholder (feed x through a placeholder),
    # rtol/atol. Asserts the two results are close.
    x_np = self._np_fft(x, rank, fft_length)
    if use_placeholder:
      x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
      x_tf = self._tf_fft(x_ph, rank, fft_length, feed_dict={x_ph: x})
    else:
      x_tf = self._tf_fft(x, rank, fft_length)

    self.assertAllClose(x_np, x_tf, rtol=rtol, atol=atol)

  def _compare_backward(self, x, rank, fft_length=None, use_placeholder=False,
                        rtol=1e-4, atol=1e-4):
    # [FUNCTION DOC]
    # HELPER: compares the TensorFlow inverse FFT of the given rank with the NumPy result.
    # Inputs: x, rank, fft_length (optional), use_placeholder, rtol/atol. Asserts the two
    # results are close.
    x_np = self._np_ifft(x, rank, fft_length)
    if use_placeholder:
      x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
      x_tf = self._tf_ifft(x_ph, rank, fft_length, feed_dict={x_ph: x})
    else:
      x_tf = self._tf_ifft(x, rank, fft_length)

    self.assertAllClose(x_np, x_tf, rtol=rtol, atol=atol)

  def _check_memory_fail(self, x, rank):
    # [FUNCTION DOC]
    # HELPER: intended for a large-batch memory-failure test.
    # KNOWN ISSUE: unused; the only test that would call it is commented out in this file.
    # Inputs: x (array) and rank. Runs the TensorFlow FFT on a GPU session limited to one percent
    # of GPU memory, expecting a memory failure to surface from the run. Returns nothing.
    config = config_pb2.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1e-2
    with self.cached_session(config=config, force_gpu=True):
      self._tf_fft(x, rank, fft_length=None)

  def _check_grad_complex(self, func, x, y, result_is_complex=True,
                          rtol=1e-2, atol=1e-2):
    # [FUNCTION DOC]
    # HELPER: numerical-versus-analytic gradient check for ops with complex input.
    # KNOWN ISSUE: the parameter result_is_complex is accepted but never used.
    # Inputs: func (callable applying an FFT variant to a complex tensor), x and y (real and
    # imaginary parts as arrays), rtol/atol. Builds a loss equal to the sum of squared
    # magnitudes of the output, computes theoretical and numerical Jacobians with respect to x
    # and y, and asserts they are close.
    with self.cached_session():

      def f(inx, iny):
        # [FUNCTION DOC]
        # HELPER: loss function for the gradient checker. Sets static shapes on both inputs,
        # combines them into a complex tensor, applies func and returns the scalar sum of
        # squared magnitudes.
        inx.set_shape(x.shape)
        iny.set_shape(y.shape)
        # func is a forward or inverse, real or complex, batched or unbatched
        # FFT function with a complex input.
        z = func(math_ops.complex(inx, iny))
        # loss = sum(|z|^2)
        loss = math_ops.reduce_sum(math_ops.real(z * math_ops.conj(z)))
        return loss

      ((x_jacob_t, y_jacob_t), (x_jacob_n, y_jacob_n)) = (
          gradient_checker_v2.compute_gradient(f, [x, y], delta=1e-2))

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=rtol, atol=atol)
    self.assertAllClose(y_jacob_t, y_jacob_n, rtol=rtol, atol=atol)

  def _check_grad_real(self, func, x, rtol=1e-2, atol=1e-2):
    # [FUNCTION DOC]
    # HELPER: numerical-versus-analytic gradient check for real-input ops (RFFT variants).
    # Inputs: func (callable taking a real tensor), x (real array), rtol/atol. Builds a loss that
    # is the sum of squared magnitudes of the output, computes theoretical and numerical
    # Jacobians and asserts they are close.
    def f(inx):
      # [FUNCTION DOC]
      # HELPER: loss function for the gradient checker. Sets the static shape, applies func and
      # returns the scalar sum of squared magnitudes of the result.
      inx.set_shape(x.shape)
      # func is a forward RFFT function (batched or unbatched).
      z = func(inx)
      # loss = sum(|z|^2)
      loss = math_ops.reduce_sum(math_ops.real(z * math_ops.conj(z)))
      return loss

    (x_jacob_t,), (x_jacob_n,) = gradient_checker_v2.compute_gradient(
        f, [x], delta=1e-2)
    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=rtol, atol=atol)


@test_util.run_all_in_graph_and_eager_modes
class FFTOpsTest(BaseFFTOpsTest, parameterized.TestCase):
  # [CLASS DOC]
  # Tests of the complex FFT and inverse FFT ops (ranks 1 to 3 and the n-dimensional versions)
  # against NumPy, including empty inputs, large batches, placeholders, random data, error
  # handling and gradients. Supplies the TensorFlow and NumPy callables that the base class uses.

  def _tf_fft(self, x, rank, fft_length=None, feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: evaluates the TensorFlow complex forward FFT of the given rank in a session.
    # fft_length is accepted but ignored; feed_dict optionally supplies placeholder values.
    # Returns the evaluated NumPy result.
    # fft_length unused for complex FFTs.
    with self.cached_session() as sess:
      return sess.run(self._tf_fft_for_rank(rank)(x), feed_dict=feed_dict)

  def _tf_ifft(self, x, rank, fft_length=None, feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: evaluates the TensorFlow complex inverse FFT of the given rank in a session.
    # fft_length is accepted but ignored; feed_dict optionally supplies placeholder values.
    # Returns the evaluated NumPy result.
    # fft_length unused for complex FFTs.
    with self.cached_session() as sess:
      return sess.run(self._tf_ifft_for_rank(rank)(x), feed_dict=feed_dict)

  def _np_fft(self, x, rank, fft_length=None):
    # [FUNCTION DOC]
    # HELPER: NumPy reference complex forward FFT over the last rank axes (1, 2 or 3), optionally
    # padded or trimmed to fft_length. Raises a value error for any other rank.
    if rank == 1:
      return np.fft.fft2(x, s=fft_length, axes=(-1,))
    elif rank == 2:
      return np.fft.fft2(x, s=fft_length, axes=(-2, -1))
    elif rank == 3:
      return np.fft.fft2(x, s=fft_length, axes=(-3, -2, -1))
    else:
      raise ValueError("invalid rank")

  def _np_ifft(self, x, rank, fft_length=None):
    # [FUNCTION DOC]
    # HELPER: NumPy reference complex inverse FFT over the last rank axes (1, 2 or 3), optionally
    # padded or trimmed to fft_length. Raises a value error for any other rank.
    if rank == 1:
      return np.fft.ifft2(x, s=fft_length, axes=(-1,))
    elif rank == 2:
      return np.fft.ifft2(x, s=fft_length, axes=(-2, -1))
    elif rank == 3:
      return np.fft.ifft2(x, s=fft_length, axes=(-3, -2, -1))
    else:
      raise ValueError("invalid rank")

  def _tf_fftn(self, x, fft_length=None, axes=None, norm=None, feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: evaluates the TensorFlow n-dimensional complex FFT in a session with the given
    # fft_length, axes and normalization; feed_dict optionally supplies placeholder values.
    # Returns the evaluated result.
    with self.cached_session() as sess:
      return sess.run(
          fft_ops.fftnd(x, fft_length=fft_length, axes=axes, norm=norm),
          feed_dict=feed_dict,
      )

  def _tf_ifftn(self, x, fft_length=None, axes=None, norm=None, feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: evaluates the TensorFlow n-dimensional complex inverse FFT in a session with the
    # given fft_length, axes and normalization; feed_dict optionally supplies placeholder values.
    # Returns the evaluated result.
    with self.cached_session() as sess:
      return sess.run(
          fft_ops.ifftnd(x, fft_length=fft_length, axes=axes, norm=norm),
          feed_dict=feed_dict,
      )

  def _np_fftn(self, x, fft_length=None, axes=None, norm=None):
    # [FUNCTION DOC]
    # HELPER: NumPy reference n-dimensional complex FFT with optional lengths, axes and norm.
    return np.fft.fftn(x, s=fft_length, axes=axes, norm=norm)

  def _np_ifftn(self, x, fft_length=None, axes=None, norm=None):
    # [FUNCTION DOC]
    # HELPER: NumPy reference n-dimensional complex inverse FFT with optional lengths, axes, norm.
    return np.fft.ifftn(x, s=fft_length, axes=axes, norm=norm)

  def _tf_fft_for_rank(self, rank):
    # [FUNCTION DOC]
    # HELPER: maps a rank (1, 2, 3) to the matching TensorFlow complex forward FFT op.
    # Raises a value error for any other rank.
    if rank == 1:
      return fft_ops.fft
    elif rank == 2:
      return fft_ops.fft2d
    elif rank == 3:
      return fft_ops.fft3d
    else:
      raise ValueError("invalid rank")

  def _tf_ifft_for_rank(self, rank):
    # [FUNCTION DOC]
    # HELPER: maps a rank (1, 2, 3) to the matching TensorFlow complex inverse FFT op.
    # Raises a value error for any other rank.
    if rank == 1:
      return fft_ops.ifft
    elif rank == 2:
      return fft_ops.ifft2d
    elif rank == 3:
      return fft_ops.ifft3d
    else:
      raise ValueError("invalid rank")

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(3), (np.complex64, np.complex128)))
  def test_empty(self, rank, extra_dims, np_type):
    # [FUNCTION DOC]
    # Parameterized over rank, extra batch dimensions and complex dtype. Feeds an array with all
    # dimensions of size zero and asserts forward and inverse FFT preserve the (empty) shape.
    dims = rank + extra_dims
    x = np.zeros((0,) * dims).astype(np_type)
    self.assertEqual(x.shape, self._tf_fft(x, rank).shape)
    self.assertEqual(x.shape, self._tf_ifft(x, rank).shape)

  @parameterized.parameters(
      itertools.product(VALID_FFT_RANKS, range(3),
                        (np.complex64, np.complex128)))
  def test_basic(self, rank, extra_dims, np_type):
    # [FUNCTION DOC]
    # Parameterized over rank, extra batch dimensions and complex dtype. Uses a deterministic
    # pattern of size four per dimension and compares forward and inverse FFT with NumPy, with a
    # tighter tolerance for double precision.
    dims = rank + extra_dims
    tol = 1e-4 if np_type == np.complex64 else 1e-8
    self._compare(
        np.mod(np.arange(np.power(4, dims)), 10).reshape(
            (4,) * dims).astype(np_type), rank, rtol=tol, atol=tol)

  @parameterized.parameters(
      itertools.product(range(3, 5), (np.complex64, np.complex128))
  )
  @test_util.run_gpu_only
  def testBasic_fftn(self, dims, np_type):
    # [FUNCTION DOC]
    # GPU-only, parameterized over three or four dimensions and complex dtype. Compares the
    # n-dimensional FFT over only the last axis with a length of four against NumPy.
    fft_length = (4,)
    axes = (-1,)
    tol = 1e-4 if np_type == np.complex64 else 1e-8
    self._Compare_fftn(
        np.mod(np.arange(np.power(4, dims)), 10)
        .reshape((4,) * dims)
        .astype(np_type),
        fft_length=fft_length,
        axes=axes,
        rtol=tol,
    )

  @parameterized.parameters(
      itertools.product(range(1, 5), (np.complex64, np.complex128))
  )
  @test_util.run_gpu_only
  def testFftLength_fftn(self, dims, np_type):
    # [FUNCTION DOC]
    # GPU-only, parameterized over one to four dimensions and complex dtype. Compares the
    # n-dimensional FFT against NumPy using explicit fft lengths that trim or pad (depending on
    # dimensionality) over matching axes.
    tol = 1e-4 if np_type == np.complex64 else 1e-8
    if dims == 1:
      fft_length = (4,)
      axes = (0,)
    elif dims == 2:
      fft_length = (2, 2)
      axes = (0, 1)
    else:
      fft_length = (6, 4, 6)
      axes = (-3, -2, -1)
    self._Compare_fftn(
        np.mod(np.arange(np.power(4, dims)), 10)
        .reshape((4,) * dims)
        .astype(np_type),
        fft_length=fft_length,
        axes=axes,
        rtol=tol,
    )

  @parameterized.parameters(
      itertools.product(range(1, 4), (np.complex64, np.complex128))
  )
  @test_util.run_gpu_only
  def testAxes_fftn(self, dims, np_type):
    # [FUNCTION DOC]
    # GPU-only, parameterized over one to three dimensions and complex dtype. Compares the
    # n-dimensional FFT against NumPy for different axis choices, including the default of no
    # explicit lengths or axes for three dimensions.
    tol = 1e-4 if np_type == np.complex64 else 1e-8
    if dims == 1:
      fft_length = (4,)
      axes = (-1,)
    elif dims == 2:
      fft_length = (4, 4)
      axes = (0, 1)
    else:
      fft_length = None
      axes = None
    self._Compare_fftn(
        np.mod(np.arange(np.power(4, dims)), 10)
        .reshape((4,) * dims)
        .astype(np_type),
        fft_length=fft_length,
        axes=axes,
        rtol=tol,
    )

  @test_util.run_gpu_only
  def testAxesError_fftn(self):
    # [FUNCTION DOC]
    # GPU-only. Asserts the n-dimensional FFT rejects invalid axes: a value error when the input
    # has too low a rank for two axes, and invalid-argument errors when the last axis is not the
    # final dimension or when axes are not successive and ascending.
    with self.assertRaisesWithPredicateMatch(
        ValueError, "Shape .* must have rank at least {}.*".format(2)
    ):
      with self.cached_session():
        self.evaluate(self._tf_fftn(np.zeros((8,)), axes=(1, 0)))

    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        "The last axis to perform transform on must be -1",
    ):
      with self.cached_session():
        self.evaluate(
            self._tf_fftn(
                np.zeros((
                    8,
                    8,
                    8,
                )),
                axes=(1,),
            )
        )

    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError, "axes must be successive and ascending."
    ):
      with self.cached_session():
        self.evaluate(self._tf_fftn(np.zeros((8, 8, 8)), axes=(1, 0, 2)))

    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError, "axes must be successive and ascending."
    ):
      with self.cached_session():
        self.evaluate(self._tf_fftn(np.zeros((8, 8, 8, 8)), axes=(0, 2, -1)))

  @parameterized.parameters(
      itertools.product(
          ("backward", "ortho", "forward"), (np.complex64, np.complex128)
      )
  )
  @test_util.run_gpu_only
  def testNorm_fftn(self, norm, np_type):
    # [FUNCTION DOC]
    # GPU-only, parameterized over the three normalization modes (backward, ortho, forward) and
    # complex dtype. Compares the n-dimensional FFT over the last two axes against NumPy.
    tol = 1e-4 if np_type == np.complex64 else 1e-8
    self._Compare_fftn(
        np.mod(np.arange(np.power(4, 4)), 10).reshape((4,) * 4).astype(np_type),
        fft_length=(4, 4),
        axes=(-2, -1),
        norm=norm,
        rtol=tol,
    )

  @parameterized.parameters(itertools.product(
      (1,), range(3), (np.complex64, np.complex128)))
  def test_large_batch(self, rank, extra_dims, np_type):
    # [FUNCTION DOC]
    # Parameterized over extra batch dimensions and complex dtype (rank one only). Uses 128
    # entries per dimension and compares forward and inverse FFT against NumPy.
    dims = rank + extra_dims
    tol = 1e-4 if np_type == np.complex64 else 5e-5
    self._compare(
        np.mod(np.arange(np.power(128, dims)), 10).reshape(
            (128,) * dims).astype(np_type), rank, rtol=tol, atol=tol)

  # TODO(yangzihao): Disable before we can figure out a way to
  # properly test memory fail for large batch fft.
  # def test_large_batch_memory_fail(self):
  #   if test.is_gpu_available(cuda_only=True):
  #     rank = 1
  #     for dims in range(rank, rank + 3):
  #       self._check_memory_fail(
  #           np.mod(np.arange(np.power(128, dims)), 64).reshape(
  #               (128,) * dims).astype(np.complex64), rank)

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(3), (np.complex64, np.complex128)))
  def test_placeholder(self, rank, extra_dims, np_type):
    # [FUNCTION DOC]
    # Parameterized over rank, extra dimensions and complex dtype. Does nothing in eager mode.
    # In graph mode, feeds the input through a placeholder of unknown shape and compares forward
    # and inverse FFT with NumPy.
    if context.executing_eagerly():
      return
    tol = 1e-4 if np_type == np.complex64 else 1e-8
    dims = rank + extra_dims
    self._compare(
        np.mod(np.arange(np.power(4, dims)), 10).reshape(
            (4,) * dims).astype(np_type),
        rank, use_placeholder=True, rtol=tol, atol=tol)

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(3), (np.complex64, np.complex128)))
  def test_random(self, rank, extra_dims, np_type):
    # [FUNCTION DOC]
    # Parameterized over rank, extra dimensions and complex dtype. Generates random complex data
    # with four entries per dimension and compares forward and inverse FFT with NumPy.
    tol = 1e-4 if np_type == np.complex64 else 5e-6
    dims = rank + extra_dims
    def gen(shape):
      # [FUNCTION DOC]
      # HELPER: returns a complex array of the given shape with uniform random real and
      # imaginary parts.
      n = np.prod(shape)
      re = np.random.uniform(size=n)
      im = np.random.uniform(size=n)
      return (re + im * 1j).reshape(shape)

    self._compare(gen((4,) * dims).astype(np_type), rank,
                  rtol=tol, atol=tol)

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS,
      # Check a variety of sizes (power-of-2, odd, etc.)
      [128, 256, 512, 1024, 127, 255, 511, 1023],
      (np.complex64, np.complex128)))
  def test_random_1d(self, rank, dim, np_type):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the rank parameter is ignored; the comparison is always rank one, so the
    # parameterization over ranks only repeats the same test.
    # Parameterized over rank, length (powers of two and odd sizes) and complex dtype. Picks a
    # tolerance depending on whether a CUDA GPU is available, then compares forward and inverse
    # one-dimensional FFT of random data with NumPy.
    has_gpu = test.is_gpu_available(cuda_only=True)
    tol = {(np.complex64, True): 1e-4,
           (np.complex64, False): 1e-2,
           (np.complex128, True): 1e-4,
           (np.complex128, False): 1e-2}[(np_type, has_gpu)]
    def gen(shape):
      # [FUNCTION DOC]
      # HELPER: returns a complex array of the given shape with uniform random real and
      # imaginary parts.
      n = np.prod(shape)
      re = np.random.uniform(size=n)
      im = np.random.uniform(size=n)
      return (re + im * 1j).reshape(shape)

    self._compare(gen((dim,)).astype(np_type), 1, rtol=tol, atol=tol)

  def test_error(self):
    # [FUNCTION DOC]
    # Does nothing in eager mode (known TODO). In graph mode, for each rank feeds inputs with
    # fewer dimensions than the rank and asserts both forward and inverse FFT raise a value
    # error about the required rank.
    # TODO(rjryan): Fix this test under Eager.
    if context.executing_eagerly():
      return
    for rank in VALID_FFT_RANKS:
      for dims in range(0, rank):
        x = np.zeros((1,) * dims).astype(np.complex64)
        with self.assertRaisesWithPredicateMatch(
            ValueError, "Shape must be .*rank {}.*".format(rank)):
          self._tf_fft(x, rank)
        with self.assertRaisesWithPredicateMatch(
            ValueError, "Shape must be .*rank {}.*".format(rank)):
          self._tf_ifft(x, rank)

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(2), (np.float32, np.float64)))
  def test_grad_simple(self, rank, extra_dims, np_type):
    # [FUNCTION DOC]
    # Parameterized over rank, extra dimensions and float dtype. Uses constant real part and zero
    # imaginary part and checks numerical against analytic gradients of the forward and inverse
    # complex FFT.
    tol = 1e-4 if np_type == np.float32 else 1e-10
    dims = rank + extra_dims
    re = np.ones(shape=(4,) * dims, dtype=np_type) / 10.0
    im = np.zeros(shape=(4,) * dims, dtype=np_type)
    self._check_grad_complex(self._tf_fft_for_rank(rank), re, im,
                             rtol=tol, atol=tol)
    self._check_grad_complex(self._tf_ifft_for_rank(rank), re, im,
                             rtol=tol, atol=tol)

  @unittest.skip("16.86% flaky")
  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(2), (np.float32, np.float64)))
  def test_grad_random(self, rank, extra_dims, np_type):
    # [FUNCTION DOC]
    # Skipped unconditionally because it is about 17 percent flaky. Parameterized over rank,
    # extra dimensions and float dtype; with random parts it would check numerical against
    # analytic gradients of the forward and inverse complex FFT.
    dims = rank + extra_dims
    tol = 1e-2 if np_type == np.float32 else 1e-10
    re = np.random.rand(*((3,) * dims)).astype(np_type) * 2 - 1
    im = np.random.rand(*((3,) * dims)).astype(np_type) * 2 - 1
    self._check_grad_complex(self._tf_fft_for_rank(rank), re, im,
                             rtol=tol, atol=tol)
    self._check_grad_complex(self._tf_ifft_for_rank(rank), re, im,
                             rtol=tol, atol=tol)


@test_util.run_all_in_graph_and_eager_modes
class RFFTOpsTest(BaseFFTOpsTest, parameterized.TestCase):
  # [CLASS DOC]
  # Tests of the real-input FFT (RFFT) and its inverse (IRFFT) for ranks 1 to 3 and the
  # n-dimensional versions against NumPy, including padding and truncation through fft_length,
  # large batches, placeholders, random data, error handling, gradients and an invalid-argument
  # regression case. Supplies the TensorFlow and NumPy callables used by the base class.

  def _tf_fft(self, x, rank, fft_length=None, feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: evaluates the TensorFlow real forward FFT of the given rank in a session using
    # fft_length; feed_dict optionally supplies placeholder values. Returns the evaluated result.
    with self.cached_session() as sess:
      return sess.run(
          self._tf_fft_for_rank(rank)(x, fft_length), feed_dict=feed_dict)

  def _tf_ifft(self, x, rank, fft_length=None, feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: evaluates the TensorFlow inverse real FFT of the given rank in a session using
    # fft_length; feed_dict optionally supplies placeholder values. Returns the evaluated result.
    with self.cached_session() as sess:
      return sess.run(
          self._tf_ifft_for_rank(rank)(x, fft_length), feed_dict=feed_dict)

  def _tf_fftn(self, x, fft_length=None, axes=None, norm=None, feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: evaluates the TensorFlow n-dimensional real FFT in a session with the given
    # fft_length, axes and normalization; feed_dict optionally supplies placeholder values.
    # Returns the evaluated result.
    with self.cached_session() as sess:
      return sess.run(
          fft_ops.rfftnd(x, fft_length=fft_length, axes=axes, norm=norm),
          feed_dict=feed_dict,
      )

  def _tf_ifftn(self, x, fft_length=None, axes=None, norm=None, feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: evaluates the TensorFlow n-dimensional inverse real FFT in a session with the given
    # fft_length, axes and normalization; feed_dict optionally supplies placeholder values.
    # Returns the evaluated result.
    with self.cached_session() as sess:
      return sess.run(
          fft_ops.irfftnd(x, fft_length=fft_length, axes=axes, norm=norm),
          feed_dict=feed_dict,
      )

  def _np_fftn(self, x, fft_length=None, axes=None, norm=None):
    # [FUNCTION DOC]
    # HELPER: NumPy reference n-dimensional real FFT with optional lengths, axes and norm.
    return np.fft.rfftn(x, s=fft_length, axes=axes, norm=norm)

  def _np_ifftn(self, x, fft_length=None, axes=None, norm=None):
    # [FUNCTION DOC]
    # HELPER: NumPy reference n-dimensional inverse real FFT with optional lengths, axes, norm.
    return np.fft.irfftn(x, s=fft_length, axes=axes, norm=norm)

  def _np_fft(self, x, rank, fft_length=None):
    # [FUNCTION DOC]
    # HELPER: NumPy reference real forward FFT over the last rank axes (1, 2 or 3), optionally
    # padded or trimmed to fft_length. Raises a value error for any other rank.
    if rank == 1:
      return np.fft.rfft2(x, s=fft_length, axes=(-1,))
    elif rank == 2:
      return np.fft.rfft2(x, s=fft_length, axes=(-2, -1))
    elif rank == 3:
      return np.fft.rfft2(x, s=fft_length, axes=(-3, -2, -1))
    else:
      raise ValueError("invalid rank")

  def _np_ifft(self, x, rank, fft_length=None):
    # [FUNCTION DOC]
    # HELPER: NumPy reference inverse real FFT over the last rank axes (1, 2 or 3), optionally
    # padded or trimmed to fft_length. Raises a value error for any other rank.
    if rank == 1:
      return np.fft.irfft2(x, s=fft_length, axes=(-1,))
    elif rank == 2:
      return np.fft.irfft2(x, s=fft_length, axes=(-2, -1))
    elif rank == 3:
      return np.fft.irfft2(x, s=fft_length, axes=(-3, -2, -1))
    else:
      raise ValueError("invalid rank")

  def _tf_fft_for_rank(self, rank):
    # [FUNCTION DOC]
    # HELPER: maps a rank (1, 2, 3) to the matching TensorFlow real forward FFT op.
    # Raises a value error for any other rank.
    if rank == 1:
      return fft_ops.rfft
    elif rank == 2:
      return fft_ops.rfft2d
    elif rank == 3:
      return fft_ops.rfft3d
    else:
      raise ValueError("invalid rank")

  def _tf_ifft_for_rank(self, rank):
    # [FUNCTION DOC]
    # HELPER: maps a rank (1, 2, 3) to the matching TensorFlow inverse real FFT op.
    # Raises a value error for any other rank.
    if rank == 1:
      return fft_ops.irfft
    elif rank == 2:
      return fft_ops.irfft2d
    elif rank == 3:
      return fft_ops.irfft3d
    else:
      raise ValueError("invalid rank")

  # rocFFT requires/assumes that the input to the irfft transform
  # is of the form that is a valid output from the rfft transform
  # (i.e. it cannot be a set of random numbers)
  # So for ROCm, call rfft and use its output as the input for testing irfft
  def _generate_valid_irfft_input(self, c2r, np_ctype, r2c, np_rtype, rank,
                                  fft_length):
    # [FUNCTION DOC]
    # HELPER: produces input for inverse real FFT tests.
    # Inputs: c2r (complex candidate data), np_ctype (complex dtype), r2c (real data), np_rtype
    # (real dtype), rank and fft_length. On ROCm builds returns the forward real FFT of r2c
    # (a valid spectrum); otherwise returns c2r cast to the complex dtype.
    if test.is_built_with_rocm():
      return self._np_fft(r2c.astype(np_rtype), rank, fft_length)
    else:
      return c2r.astype(np_ctype)

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(3), (np.float32, np.float64)))

  def test_empty(self, rank, extra_dims, np_rtype):
    # [FUNCTION DOC]
    # Parameterized over rank, extra dimensions and real dtype. Asserts the real forward FFT of
    # an empty real array and the inverse FFT of an empty complex array preserve the shape.
    np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
    dims = rank + extra_dims
    x = np.zeros((0,) * dims).astype(np_rtype)
    self.assertEqual(x.shape, self._tf_fft(x, rank).shape)
    x = np.zeros((0,) * dims).astype(np_ctype)
    self.assertEqual(x.shape, self._tf_ifft(x, rank).shape)

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(3), (5, 6), (np.float32, np.float64)))
  def test_basic(self, rank, extra_dims, size, np_rtype):
    # [FUNCTION DOC]
    # Parameterized over rank, extra dimensions, size (5 or 6) and real dtype. Compares the real
    # forward FFT of a deterministic pattern, and the inverse FFT of a matching half-spectrum
    # input, against NumPy with fft_length equal to the size.
    np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
    tol = 1e-4 if np_rtype == np.float32 else 5e-5
    dims = rank + extra_dims
    inner_dim = size // 2 + 1
    r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape(
        (size,) * dims)
    fft_length = (size,) * rank
    self._compare_forward(
        r2c.astype(np_rtype), rank, fft_length, rtol=tol, atol=tol)
    c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim),
                 10).reshape((size,) * (dims - 1) + (inner_dim,))
    c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, rank,
                                           fft_length)
    self._compare_backward(c2r, rank, fft_length, rtol=tol, atol=tol)

  @parameterized.parameters(
      itertools.product(range(3, 5), (5, 6), (np.float32, np.float64))
  )
  @test_util.run_gpu_only
  def testBasic_rfftn(self, dims, size, np_rtype):
    # [FUNCTION DOC]
    # GPU-only, parameterized over dimensions three or four, size and real dtype. Compares the
    # n-dimensional real forward and inverse FFT over the last two axes against NumPy.
    fft_length = (size, size)
    axes = (-2, -1)
    inner_dim = size // 2 + 1
    np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
    tol = 1e-4 if np_ctype == np.complex64 else 1e-8
    r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape((size,) * dims)
    self._CompareForward_fftn(
        r2c.astype(np_rtype),
        fft_length=fft_length,
        axes=axes,
        rtol=tol,
    )
    c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim), 10).reshape(
        (size,) * (dims - 1) + (inner_dim,)
    )
    c2r = self._generate_valid_irfft_input(
        c2r, np_ctype, r2c, np_rtype, 2, fft_length
    )
    self._CompareBackward_fftn(c2r, fft_length, axes, rtol=tol)

  @parameterized.parameters(
      itertools.product(range(1, 5), (5, 6), (np.float32, np.float64))
  )
  @test_util.run_gpu_only
  def testFftLength_rfftn(self, dims, size, np_rtype):
    # [FUNCTION DOC]
    # GPU-only, parameterized over one to four dimensions, size and real dtype. Compares the
    # n-dimensional real forward and inverse FFT against NumPy with explicit trimming or padding
    # lengths depending on dimensionality.
    inner_dim = size // 2 + 1
    np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
    tol = 1e-4 if np_ctype == np.complex64 else 1e-8
    r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape((size,) * dims)
    if dims == 1:
      fft_length = (size,)
      axes = (-1,)
    elif dims == 2:
      fft_length = (size // 2, size // 2)
      axes = (-2, -1)
    else:
      fft_length = (size * 2, size, size * 2)
      axes = (-3, -2, -1)
    self._CompareForward_fftn(
        r2c.astype(np_rtype),
        fft_length=fft_length,
        axes=axes,
        rtol=tol,
    )
    c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim), 10).reshape(
        (size,) * (dims - 1) + (inner_dim,)
    )
    c2r = self._generate_valid_irfft_input(
        c2r, np_ctype, r2c, np_rtype, 2, fft_length
    )
    self._CompareBackward_fftn(c2r, fft_length, axes, rtol=tol)

  @parameterized.parameters(
      itertools.product(range(1, 4), (5, 6), (np.float32, np.float64))
  )
  @test_util.run_gpu_only
  def testAxes_rfftn(self, dims, size, np_rtype):
    # [FUNCTION DOC]
    # GPU-only, parameterized over one to three dimensions, size and real dtype. Compares the
    # n-dimensional real forward and inverse FFT against NumPy for different axes choices,
    # including no explicit lengths or axes for three dimensions.
    inner_dim = size // 2 + 1
    np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
    tol = 1e-4 if np_ctype == np.complex64 else 1e-8
    r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape((size,) * dims)
    if dims == 1:
      fft_length = (size,)
      axes = (-1,)
    elif dims == 2:
      fft_length = (size, size)
      axes = (0, 1)
    else:
      fft_length = None
      axes = None
    self._CompareForward_fftn(
        r2c.astype(np_rtype),
        fft_length=fft_length,
        axes=axes,
        rtol=tol,
    )
    c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim), 10).reshape(
        (size,) * (dims - 1) + (inner_dim,)
    )
    c2r = self._generate_valid_irfft_input(
        c2r, np_ctype, r2c, np_rtype, 2, fft_length
    )
    self._CompareBackward_fftn(c2r, fft_length, axes, rtol=tol)

  @parameterized.parameters(
      itertools.product(
          ("backward", "ortho", "forward"), (np.float32, np.float64)
      )
  )
  @test_util.run_gpu_only
  def testNorm_rfftn(self, norm, np_rtype):
    # [FUNCTION DOC]
    # GPU-only, parameterized over the three normalization modes and real dtype. Compares the
    # n-dimensional real forward and inverse FFT over the last two axes against NumPy.
    inner_dim = 3
    np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
    tol = 1e-4 if np_ctype == np.complex64 else 1e-8
    r2c = np.mod(np.arange(np.power(5, 4)), 10).reshape((5,) * 4)
    fft_length = (5, 5)
    axes = (-2, -1)

    self._CompareForward_fftn(
        r2c.astype(np_rtype),
        fft_length=fft_length,
        axes=axes,
        norm=norm,
        rtol=tol,
    )
    c2r = np.mod(np.arange(np.power(5, 4 - 1) * inner_dim), 10).reshape(
        (5,) * (4 - 1) + (inner_dim,)
    )
    c2r = self._generate_valid_irfft_input(
        c2r, np_ctype, r2c, np_rtype, 2, fft_length
    )
    self._CompareBackward_fftn(c2r, fft_length, axes, norm=norm, rtol=tol)

  @parameterized.parameters(itertools.product(
      (1,), range(3), (64, 128), (np.float32, np.float64)))
  def test_large_batch(self, rank, extra_dims, size, np_rtype):
    # [FUNCTION DOC]
    # Parameterized over extra batch dimensions, size (64 or 128) and real dtype (rank one only).
    # Compares real forward FFT and inverse FFT of a large batch against NumPy.
    np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
    tol = 1e-4 if np_rtype == np.float32 else 1e-5
    dims = rank + extra_dims
    inner_dim = size // 2 + 1
    r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape(
        (size,) * dims)
    fft_length = (size,) * rank
    self._compare_forward(
        r2c.astype(np_rtype), rank, fft_length, rtol=tol, atol=tol)
    c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim),
                 10).reshape((size,) * (dims - 1) + (inner_dim,))
    c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, rank,
                                           fft_length)
    self._compare_backward(c2r, rank, fft_length, rtol=tol, atol=tol)

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(3), (5, 6), (np.float32, np.float64)))
  def test_placeholder(self, rank, extra_dims, size, np_rtype):
    # [FUNCTION DOC]
    # Does nothing in eager mode. In graph mode feeds data through placeholders of unknown shape
    # and compares real forward and inverse FFT against NumPy. Parameterized over rank, extra
    # dimensions, size and real dtype.
    if context.executing_eagerly():
      return
    np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
    tol = 1e-4 if np_rtype == np.float32 else 1e-8
    dims = rank + extra_dims
    inner_dim = size // 2 + 1
    r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape(
        (size,) * dims)
    fft_length = (size,) * rank
    self._compare_forward(
        r2c.astype(np_rtype),
        rank,
        fft_length,
        use_placeholder=True,
        rtol=tol,
        atol=tol)
    c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim),
                 10).reshape((size,) * (dims - 1) + (inner_dim,))
    c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, rank,
                                           fft_length)
    self._compare_backward(
        c2r, rank, fft_length, use_placeholder=True, rtol=tol, atol=tol)

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(3), (5, 6), (np.float32, np.float64)))
  def test_fft_lenth_truncate(self, rank, extra_dims, size, np_rtype):
    # [FUNCTION DOC]
    # (Name misspells "length".) Parameterized over rank, extra dimensions, size and real dtype.
    # Skipped on ROCm for rank three. Uses an fft length smaller than the data by two, compares
    # forward and inverse real FFT against NumPy, and in graph mode repeats with placeholders.
    """Test truncation (FFT size < dimensions)."""
    if test.is_built_with_rocm() and (rank == 3):
      # TODO(rocm): fix me
      # rfft fails for rank == 3 on ROCm
      self.skipTest("Test fails on ROCm...fix me")
    np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
    tol = 1e-4 if np_rtype == np.float32 else 8e-5
    dims = rank + extra_dims
    inner_dim = size // 2 + 1
    r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape(
        (size,) * dims)
    c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim),
                 10).reshape((size,) * (dims - 1) + (inner_dim,))
    fft_length = (size - 2,) * rank
    self._compare_forward(r2c.astype(np_rtype), rank, fft_length,
                          rtol=tol, atol=tol)
    c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, rank,
                                           fft_length)
    self._compare_backward(c2r, rank, fft_length, rtol=tol, atol=tol)
    # Confirm it works with unknown shapes as well.
    if not context.executing_eagerly():
      self._compare_forward(
          r2c.astype(np_rtype),
          rank,
          fft_length,
          use_placeholder=True,
          rtol=tol, atol=tol)
      self._compare_backward(
          c2r, rank, fft_length, use_placeholder=True, rtol=tol, atol=tol)

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(3), (5, 6), (np.float32, np.float64)))
  def test_fft_lenth_pad(self, rank, extra_dims, size, np_rtype):
    # [FUNCTION DOC]
    # (Name misspells "length".) Parameterized over rank, extra dimensions, size and real dtype.
    # Uses an fft length larger than the data by two, compares forward and inverse real FFT
    # against NumPy, and in graph mode repeats with placeholders.
    """Test padding (FFT size > dimensions)."""
    np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
    tol = 1e-4 if np_rtype == np.float32 else 8e-5
    dims = rank + extra_dims
    inner_dim = size // 2 + 1
    r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape(
        (size,) * dims)
    c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim),
                 10).reshape((size,) * (dims - 1) + (inner_dim,))
    fft_length = (size + 2,) * rank
    self._compare_forward(r2c.astype(np_rtype), rank, fft_length,
                          rtol=tol, atol=tol)
    c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, rank,
                                           fft_length)
    self._compare_backward(c2r.astype(np_ctype), rank, fft_length,
                           rtol=tol, atol=tol)
    # Confirm it works with unknown shapes as well.
    if not context.executing_eagerly():
      self._compare_forward(
          r2c.astype(np_rtype),
          rank,
          fft_length,
          use_placeholder=True,
          rtol=tol, atol=tol)
      self._compare_backward(
          c2r.astype(np_ctype),
          rank,
          fft_length,
          use_placeholder=True,
          rtol=tol, atol=tol)

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(3), (5, 6), (np.float32, np.float64)))
  def test_random(self, rank, extra_dims, size, np_rtype):
    # [FUNCTION DOC]
    # Parameterized over rank, extra dimensions, size and real dtype. Generates random real data
    # and random complex half-spectrum data and compares real forward and inverse FFT against
    # NumPy.
    def gen_real(shape):
      # [FUNCTION DOC]
      # HELPER: returns a real array of the given shape with uniform random values.
      n = np.prod(shape)
      re = np.random.uniform(size=n)
      ret = re.reshape(shape)
      return ret

    def gen_complex(shape):
      # [FUNCTION DOC]
      # HELPER: returns a complex array of the given shape with uniform random real and
      # imaginary parts.
      n = np.prod(shape)
      re = np.random.uniform(size=n)
      im = np.random.uniform(size=n)
      ret = (re + im * 1j).reshape(shape)
      return ret
    np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
    tol = 1e-4 if np_rtype == np.float32 else 1e-5
    dims = rank + extra_dims
    r2c = gen_real((size,) * dims)
    inner_dim = size // 2 + 1
    fft_length = (size,) * rank
    self._compare_forward(
        r2c.astype(np_rtype), rank, fft_length, rtol=tol, atol=tol)
    complex_dims = (size,) * (dims - 1) + (inner_dim,)
    c2r = gen_complex(complex_dims)
    c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, rank,
                                           fft_length)
    self._compare_backward(c2r, rank, fft_length, rtol=tol, atol=tol)

  def test_error(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the second inner loop over dims the loop variable is never used (the input
    # is always built from the rank), so both iterations run the identical checks.
    # Does nothing in eager mode (known TODO). In graph mode asserts value errors for: too few
    # dimensions, an fft_length that is not rank one, and an fft_length of the wrong length; and
    # asserts invalid-argument errors when the raw real and inverse real kernels are called on
    # inputs smaller than the requested fft_length.
    # TODO(rjryan): Fix this test under Eager.
    if context.executing_eagerly():
      return
    for rank in VALID_FFT_RANKS:
      for dims in range(0, rank):
        x = np.zeros((1,) * dims).astype(np.complex64)
        with self.assertRaisesWithPredicateMatch(
            ValueError, "Shape .* must have rank at least {}".format(rank)):
          self._tf_fft(x, rank)
        with self.assertRaisesWithPredicateMatch(
            ValueError, "Shape .* must have rank at least {}".format(rank)):
          self._tf_ifft(x, rank)
      for dims in range(rank, rank + 2):
        x = np.zeros((1,) * rank)

        # Test non-rank-1 fft_length produces an error.
        fft_length = np.zeros((1, 1)).astype(np.int32)
        with self.assertRaisesWithPredicateMatch(ValueError,
                                                 "Shape .* must have rank 1"):
          self._tf_fft(x, rank, fft_length)
        with self.assertRaisesWithPredicateMatch(ValueError,
                                                 "Shape .* must have rank 1"):
          self._tf_ifft(x, rank, fft_length)

        # Test wrong fft_length length.
        fft_length = np.zeros((rank + 1,)).astype(np.int32)
        with self.assertRaisesWithPredicateMatch(
            ValueError, "Dimension must be .*but is {}.*".format(rank + 1)):
          self._tf_fft(x, rank, fft_length)
        with self.assertRaisesWithPredicateMatch(
            ValueError, "Dimension must be .*but is {}.*".format(rank + 1)):
          self._tf_ifft(x, rank, fft_length)

      # Test that calling the kernel directly without padding to fft_length
      # produces an error.
      rffts_for_rank = {
          1: [gen_spectral_ops.rfft, gen_spectral_ops.irfft],
          2: [gen_spectral_ops.rfft2d, gen_spectral_ops.irfft2d],
          3: [gen_spectral_ops.rfft3d, gen_spectral_ops.irfft3d]
      }
      rfft_fn, irfft_fn = rffts_for_rank[rank]
      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          "Input dimension .* must have length of at least 6 but got: 5"):
        x = np.zeros((5,) * rank).astype(np.float32)
        fft_length = [6] * rank
        with self.cached_session():
          self.evaluate(rfft_fn(x, fft_length))

      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          "Input dimension .* must have length of at least .* but got: 3"):
        x = np.zeros((3,) * rank).astype(np.complex64)
        fft_length = [6] * rank
        with self.cached_session():
          self.evaluate(irfft_fn(x, fft_length))

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(2), (5, 6), (np.float32, np.float64)))
  def test_grad_simple(self, rank, extra_dims, size, np_rtype):
    # [FUNCTION DOC]
    # Parameterized over rank, extra dimensions, size and real dtype. Returns early for rank
    # three (no gradients). Checks numerical against analytic gradients of the real forward FFT
    # and, except on ROCm, the inverse real FFT using constant inputs.
    # rfft3d/irfft3d do not have gradients yet.
    if rank == 3:
      return
    dims = rank + extra_dims
    tol = 1e-3 if np_rtype == np.float32 else 1e-10
    re = np.ones(shape=(size,) * dims, dtype=np_rtype)
    im = -np.ones(shape=(size,) * dims, dtype=np_rtype)
    self._check_grad_real(self._tf_fft_for_rank(rank), re,
                          rtol=tol, atol=tol)
    if test.is_built_with_rocm():
      # Fails on ROCm because of irfft peculairity
      return
    self._check_grad_complex(
        self._tf_ifft_for_rank(rank), re, im, result_is_complex=False,
        rtol=tol, atol=tol)

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(2), (5, 6), (np.float32, np.float64)))
  def test_grad_random(self, rank, extra_dims, size, np_rtype):
    # [FUNCTION DOC]
    # Same as the simple gradient test but with random inputs in the range minus one to one.
    # Returns early for rank three and skips the inverse check on ROCm.
    # rfft3d/irfft3d do not have gradients yet.
    if rank == 3:
      return
    dims = rank + extra_dims
    tol = 1e-2 if np_rtype == np.float32 else 1e-10
    re = np.random.rand(*((size,) * dims)).astype(np_rtype) * 2 - 1
    im = np.random.rand(*((size,) * dims)).astype(np_rtype) * 2 - 1
    self._check_grad_real(self._tf_fft_for_rank(rank), re,
                          rtol=tol, atol=tol)
    if test.is_built_with_rocm():
      # Fails on ROCm because of irfft peculairity
      return
    self._check_grad_complex(
        self._tf_ifft_for_rank(rank), re, im, result_is_complex=False,
        rtol=tol, atol=tol)

  def test_invalid_args(self):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Calls the two-dimensional real FFT on an empty-column
    # array with a negative fft length and asserts a value error or invalid-argument error
    # saying the length must be non-negative.
    # Test case for GitHub issue 55263
    a = np.empty([6, 0])
    b = np.array([1, -1])
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "(.*must be greater or equal to.*)|(must >= 0)",
    ):
      with self.session():
        v = fft_ops.rfft2d(input_tensor=a, fft_length=b)
        self.evaluate(v)


@test_util.run_all_in_graph_and_eager_modes
class FFTShiftTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests of fftshift and ifftshift: exact results on small lists, axes keyword handling,
  # agreement with NumPy, unknown-shape placeholders and negative axes. Runs in graph and eager.

  def test_definition(self):
    # [FUNCTION DOC]
    # Checks fftshift and ifftshift on odd and even length frequency lists against hand-written
    # expected arrays, and that ifftshift undoes fftshift.
    with self.session():
      x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
      y = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
      self.assertAllEqual(fft_ops.fftshift(x), y)
      self.assertAllEqual(fft_ops.ifftshift(y), x)
      x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
      y = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
      self.assertAllEqual(fft_ops.fftshift(x), y)
      self.assertAllEqual(fft_ops.ifftshift(y), x)

  def test_axes_keyword(self):
    # [FUNCTION DOC]
    # Checks both shift functions on a three by three input with the axes argument given as a
    # tuple, as a single integer, and omitted (all axes), against hand-written expected values.
    with self.session():
      freqs = [[0, 1, 2], [3, 4, -4], [-3, -2, -1]]
      shifted = [[-1, -3, -2], [2, 0, 1], [-4, 3, 4]]
      self.assertAllEqual(fft_ops.fftshift(freqs, axes=(0, 1)), shifted)
      self.assertAllEqual(
          fft_ops.fftshift(freqs, axes=0),
          fft_ops.fftshift(freqs, axes=(0,)))
      self.assertAllEqual(fft_ops.ifftshift(shifted, axes=(0, 1)), freqs)
      self.assertAllEqual(
          fft_ops.ifftshift(shifted, axes=0),
          fft_ops.ifftshift(shifted, axes=(0,)))
      self.assertAllEqual(fft_ops.fftshift(freqs), shifted)
      self.assertAllEqual(fft_ops.ifftshift(shifted), freqs)

  def test_numpy_compatibility(self):
    # [FUNCTION DOC]
    # Asserts fftshift and ifftshift equal the NumPy functions on one-dimensional odd and even
    # lists and on a two-dimensional input with explicit axes.
    with self.session():
      x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
      y = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
      self.assertAllEqual(fft_ops.fftshift(x), np.fft.fftshift(x))
      self.assertAllEqual(fft_ops.ifftshift(y), np.fft.ifftshift(y))
      x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
      y = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
      self.assertAllEqual(fft_ops.fftshift(x), np.fft.fftshift(x))
      self.assertAllEqual(fft_ops.ifftshift(y), np.fft.ifftshift(y))
      freqs = [[0, 1, 2], [3, 4, -4], [-3, -2, -1]]
      shifted = [[-1, -3, -2], [2, 0, 1], [-4, 3, 4]]
      self.assertAllEqual(
          fft_ops.fftshift(freqs, axes=(0, 1)),
          np.fft.fftshift(freqs, axes=(0, 1)))
      self.assertAllEqual(
          fft_ops.ifftshift(shifted, axes=(0, 1)),
          np.fft.ifftshift(shifted, axes=(0, 1)))

  @parameterized.parameters(None, 1, ([1, 2],))
  def test_placeholder(self, axes):
    # [FUNCTION DOC]
    # Parameterized over axes (none, an integer, a list). Does nothing in eager mode. In graph
    # mode applies both shifts to a placeholder with fully unknown dimensions, feeds random data
    # and compares the results with NumPy.
    if context.executing_eagerly():
      return
    x = array_ops.placeholder(shape=[None, None, None], dtype="float32")
    y_fftshift = fft_ops.fftshift(x, axes=axes)
    y_ifftshift = fft_ops.ifftshift(x, axes=axes)
    x_np = np.random.rand(16, 256, 256)
    with self.session() as sess:
      y_fftshift_res, y_ifftshift_res = sess.run(
          [y_fftshift, y_ifftshift],
          feed_dict={x: x_np})
    self.assertAllClose(y_fftshift_res, np.fft.fftshift(x_np, axes=axes))
    self.assertAllClose(y_ifftshift_res, np.fft.ifftshift(x_np, axes=axes))

  def test_negative_axes(self):
    # [FUNCTION DOC]
    # Checks both shift functions accept negative axes values, matching hand-written expected
    # values and the equivalent positive axes.
    with self.session():
      freqs = [[0, 1, 2], [3, 4, -4], [-3, -2, -1]]
      shifted = [[-1, -3, -2], [2, 0, 1], [-4, 3, 4]]
      self.assertAllEqual(fft_ops.fftshift(freqs, axes=(0, -1)), shifted)
      self.assertAllEqual(fft_ops.ifftshift(shifted, axes=(0, -1)), freqs)
      self.assertAllEqual(
          fft_ops.fftshift(freqs, axes=-1), fft_ops.fftshift(freqs, axes=(1,)))
      self.assertAllEqual(
          fft_ops.ifftshift(shifted, axes=-1),
          fft_ops.ifftshift(shifted, axes=(1,)))


if __name__ == "__main__":
  test.main()
