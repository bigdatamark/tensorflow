# [MODULE DOC]
# Tests for Fourier transforms (1D, 2D, 3D, real and inverse real variants) and the short-time
# Fourier transform on XLA devices. Defines tolerance constants, test size lists built at import
# time (random shuffles seeded for determinism, sizes limited to avoid memory exhaustion on GPU),
# small numpy conversion helpers, and one test class that compares results with numpy or scipy.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for FFT via the XLA JIT."""

import itertools

import numpy as np
import scipy.signal as sps

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops.signal import signal
from tensorflow.python.platform import googletest

BATCH_DIMS = (3, 5)
RTOL = 0.009  # Eigen/cuFFT differ widely from np, especially for FFT3D
ATOL = 1e-4
RTOL_3D = 0.07
ATOL_3D = 4e-4


def pick_10(x):
  # [FUNCTION DOC]
  # HELPER: picks test sizes for the multi-dimensional FFT tests.
  # Takes any iterable, turns it into a list, shuffles it with a fixed random seed (resetting the
  # global numpy seed as a side effect) and returns the first ten items, or all if fewer.
  x = list(x)
  np.random.seed(123)
  np.random.shuffle(x)
  return x[:10]


def to_32bit(x):
  # [FUNCTION DOC]
  # HELPER: narrows numpy data to 32-bit precision for comparison.
  # Converts 128-bit complex arrays to 64-bit complex and 64-bit floats to 32-bit floats; any other
  # array is returned unchanged.
  if x.dtype == np.complex128:
    return x.astype(np.complex64)
  if x.dtype == np.float64:
    return x.astype(np.float32)
  return x


POWS_OF_2 = 2**np.arange(3, 12)
INNER_DIMS_1D = list((x,) for x in POWS_OF_2)
POWS_OF_2 = 2**np.arange(3, 8)  # To avoid OOM on GPU.
INNER_DIMS_2D = pick_10(itertools.product(POWS_OF_2, POWS_OF_2))
INNER_DIMS_3D = pick_10(itertools.product(POWS_OF_2, POWS_OF_2, POWS_OF_2))


class FFTTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Compares FFT-family ops run in the XLA test scope against numpy and scipy references, using a
  # shared verification helper with per-dimension tolerances.

  def _VerifyFftMethod(self,
                       inner_dims,
                       complex_to_input,
                       input_to_expected,
                       tf_method,
                       atol=ATOL,
                       rtol=RTOL):
    # [FUNCTION DOC]
    # HELPER: shared by all FFT tests.
    # Takes the list of inner dimension tuples to test, a function turning complex test data into
    # the op input, a function computing the expected result from that input, the op under test, and
    # optional absolute and relative tolerances (defaults are the module constants). For each inner
    # dimension it builds shuffled complex data with batch dimensions (3, 5) (seeding the global
    # numpy generator), converts to 32 bit, prints the size, runs the op on a placeholder in a
    # session and asserts closeness to the expected value.
    for indims in inner_dims:
      print("nfft =", indims)
      shape = BATCH_DIMS + indims
      data = np.arange(np.prod(shape) * 2) / np.prod(indims)
      np.random.seed(123)
      np.random.shuffle(data)
      data = np.reshape(data.astype(np.float32).view(np.complex64), shape)
      data = to_32bit(complex_to_input(data))
      expected = to_32bit(input_to_expected(data))
      with self.session() as sess:
        with self.test_scope():
          ph = array_ops.placeholder(
              dtypes.as_dtype(data.dtype), shape=data.shape)
          out = tf_method(ph)
        value = sess.run(out, {ph: data})
        self.assertAllClose(expected, value, rtol=rtol, atol=atol)

  def testContribSignalSTFT(self):
    # [FUNCTION DOC]
    # Computes the STFT (window 512, hop 128) of random-ordered real data of length 10240 with
    # batch dimensions, and compares with scipy using a Hann window (swapping axes and scaling by
    # the window sum, since scipy normalizes). Also builds the gradient and only checks it runs.
    ws = 512
    hs = 128
    dims = (ws * 20,)
    shape = BATCH_DIMS + dims
    data = np.arange(np.prod(shape)) / np.prod(dims)
    np.random.seed(123)
    np.random.shuffle(data)
    data = np.reshape(data.astype(np.float32), shape)
    window = sps.get_window("hann", ws)
    expected = sps.stft(
        data, nperseg=ws, noverlap=ws - hs, boundary=None, window=window)[2]
    expected = np.swapaxes(expected, -1, -2)
    expected *= window.sum()  # scipy divides by window sum
    with self.session() as sess:
      with self.test_scope():
        ph = array_ops.placeholder(
            dtypes.as_dtype(data.dtype), shape=data.shape)
        out = signal.stft(ph, ws, hs)
        grad = gradients_impl.gradients(out, ph,
                                        grad_ys=array_ops.ones_like(out))

      # For gradients, we simply verify that they compile & execute.
      value, _ = sess.run([out, grad], {ph: data})
      self.assertAllClose(expected, value, rtol=RTOL, atol=ATOL)

  def testFFT(self):
    # [FUNCTION DOC]
    # 1D complex FFT versus numpy for sizes 8 to 2048.
    self._VerifyFftMethod(INNER_DIMS_1D, lambda x: x, np.fft.fft,
                          signal.fft)

  def testFFT2D(self):
    # [FUNCTION DOC]
    # 2D complex FFT versus numpy for ten sampled size pairs.
    self._VerifyFftMethod(INNER_DIMS_2D, lambda x: x, np.fft.fft2,
                          signal.fft2d)

  def testFFT3D(self):
    # [FUNCTION DOC]
    # 3D complex FFT versus numpy over the last three axes, with looser 3D tolerances.
    self._VerifyFftMethod(INNER_DIMS_3D, lambda x: x,
                          lambda x: np.fft.fftn(x, axes=(-3, -2, -1)),
                          signal.fft3d, ATOL_3D, RTOL_3D)

  def testIFFT(self):
    # [FUNCTION DOC]
    # 1D inverse complex FFT versus numpy.
    self._VerifyFftMethod(INNER_DIMS_1D, lambda x: x, np.fft.ifft,
                          signal.ifft)

  def testIFFT2D(self):
    # [FUNCTION DOC]
    # 2D inverse complex FFT versus numpy.
    self._VerifyFftMethod(INNER_DIMS_2D, lambda x: x, np.fft.ifft2,
                          signal.ifft2d)

  def testIFFT3D(self):
    # [FUNCTION DOC]
    # 3D inverse complex FFT versus numpy over the last three axes, with looser tolerances.
    self._VerifyFftMethod(INNER_DIMS_3D, lambda x: x,
                          lambda x: np.fft.ifftn(x, axes=(-3, -2, -1)),
                          signal.ifft3d, ATOL_3D, RTOL_3D)

  def testRFFT(self):
    # [FUNCTION DOC]
    # 1D real FFT versus numpy on the real part of the data, using the last dimension as length.

    def _to_expected(x):
      # [FUNCTION DOC]
      # HELPER: expected result using numpy real FFT with length equal to the last dimension.
      return np.fft.rfft(x, n=x.shape[-1])

    def _tf_fn(x):
      # [FUNCTION DOC]
      # HELPER: the op under test, real FFT with length equal to the last dimension.
      return signal.rfft(x, fft_length=[x.shape[-1]])

    self._VerifyFftMethod(INNER_DIMS_1D, np.real, _to_expected, _tf_fn)

  def testRFFT2D(self):
    # [FUNCTION DOC]
    # 2D real FFT versus numpy on the real part, using the last two dimensions as the length.

    def _tf_fn(x):
      # [FUNCTION DOC]
      # HELPER: the op under test, 2D real FFT over the last two dimensions.
      return signal.rfft2d(x, fft_length=[x.shape[-2], x.shape[-1]])

    self._VerifyFftMethod(
        INNER_DIMS_2D, np.real,
        lambda x: np.fft.rfft2(x, s=[x.shape[-2], x.shape[-1]]), _tf_fn)

  def testRFFT3D(self):
    # [FUNCTION DOC]
    # 3D real FFT versus numpy on the real part, with looser 3D tolerances.

    def _to_expected(x):
      # [FUNCTION DOC]
      # HELPER: expected result using numpy n-dimensional real FFT over the last three axes.
      return np.fft.rfftn(
          x, axes=(-3, -2, -1), s=[x.shape[-3], x.shape[-2], x.shape[-1]])

    def _tf_fn(x):
      # [FUNCTION DOC]
      # HELPER: the op under test, 3D real FFT over the last three dimensions.
      return signal.rfft3d(
          x, fft_length=[x.shape[-3], x.shape[-2], x.shape[-1]])

    self._VerifyFftMethod(INNER_DIMS_3D, np.real, _to_expected, _tf_fn, ATOL_3D,
                          RTOL_3D)

  def testRFFT3DMismatchedSize(self):
    # [FUNCTION DOC]
    # 3D real FFT where the requested length differs from the data (first axis halved, last axis
    # doubled), so the data is cropped or zero-padded; compared with numpy using default tolerances.

    def _to_expected(x):
      # [FUNCTION DOC]
      # HELPER: expected result from numpy with the mismatched lengths.
      return np.fft.rfftn(
          x,
          axes=(-3, -2, -1),
          s=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])

    def _tf_fn(x):
      # [FUNCTION DOC]
      # HELPER: the op under test with the mismatched lengths.
      return signal.rfft3d(
          x, fft_length=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])

    self._VerifyFftMethod(INNER_DIMS_3D, np.real, _to_expected, _tf_fn)

  def testIRFFT(self):
    # [FUNCTION DOC]
    # 1D inverse real FFT versus numpy; input is the real FFT of the real part, output length is
    # twice the input's last dimension minus two.

    def _tf_fn(x):
      # [FUNCTION DOC]
      # HELPER: the op under test, inverse real FFT producing an even output length.
      return signal.irfft(x, fft_length=[2 * (x.shape[-1] - 1)])

    self._VerifyFftMethod(
        INNER_DIMS_1D, lambda x: np.fft.rfft(np.real(x), n=x.shape[-1]),
        lambda x: np.fft.irfft(x, n=2 * (x.shape[-1] - 1)), _tf_fn)

  def testIRFFT2D(self):
    # [FUNCTION DOC]
    # 2D inverse real FFT versus numpy, with the output last dimension doubled minus two.

    def _tf_fn(x):
      # [FUNCTION DOC]
      # HELPER: the op under test, 2D inverse real FFT.
      return signal.irfft2d(x, fft_length=[x.shape[-2], 2 * (x.shape[-1] - 1)])

    self._VerifyFftMethod(
        INNER_DIMS_2D,
        lambda x: np.fft.rfft2(np.real(x), s=[x.shape[-2], x.shape[-1]]),
        lambda x: np.fft.irfft2(x, s=[x.shape[-2], 2 * (x.shape[-1] - 1)]),
        _tf_fn)

  def testIRFFT3D(self):
    # [FUNCTION DOC]
    # 3D inverse real FFT versus numpy using looser 3D tolerances; three helpers build the input,
    # expected output and op result.

    def _to_input(x):
      # [FUNCTION DOC]
      # HELPER: builds the op input as the numpy 3D real FFT of the real part of the data.
      return np.fft.rfftn(
          np.real(x),
          axes=(-3, -2, -1),
          s=[x.shape[-3], x.shape[-2], x.shape[-1]])

    def _to_expected(x):
      # [FUNCTION DOC]
      # HELPER: expected result using numpy 3D inverse real FFT with an even last-axis output length.
      return np.fft.irfftn(
          x,
          axes=(-3, -2, -1),
          s=[x.shape[-3], x.shape[-2], 2 * (x.shape[-1] - 1)])

    def _tf_fn(x):
      # [FUNCTION DOC]
      # HELPER: the op under test, 3D inverse real FFT.
      return signal.irfft3d(
          x, fft_length=[x.shape[-3], x.shape[-2], 2 * (x.shape[-1] - 1)])

    self._VerifyFftMethod(INNER_DIMS_3D, _to_input, _to_expected, _tf_fn,
                          ATOL_3D, RTOL_3D)

  def testIRFFT3DMismatchedSize(self):
    # [FUNCTION DOC]
    # 3D inverse real FFT where the transform lengths differ from the data (first axis halved, last
    # axis doubled); compared with numpy using the looser 3D tolerances.

    def _to_input(x):
      # [FUNCTION DOC]
      # HELPER: builds the op input as a numpy 3D real FFT with the mismatched lengths.
      return np.fft.rfftn(
          np.real(x),
          axes=(-3, -2, -1),
          s=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])

    def _to_expected(x):
      # [FUNCTION DOC]
      # HELPER: expected result from numpy 3D inverse real FFT with the mismatched lengths.
      return np.fft.irfftn(
          x,
          axes=(-3, -2, -1),
          s=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])

    def _tf_fn(x):
      # [FUNCTION DOC]
      # HELPER: the op under test with the mismatched lengths.
      return signal.irfft3d(
          x, fft_length=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])

    self._VerifyFftMethod(INNER_DIMS_3D, _to_input, _to_expected, _tf_fn,
                          ATOL_3D, RTOL_3D)



if __name__ == "__main__":
  googletest.main()
