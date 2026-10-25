# [MODULE DOC]
# Unit tests for the window functions in the signal ops (Kaiser, Hann, Hamming, Vorbis and
# Kaiser-Bessel-derived windows). Compares against NumPy/SciPy-style reference implementations
# across window lengths, periodic/symmetric variants and float16/32/64 tolerances, checks MDCT
# window conditions (Princen-Bradley and symmetry) and that windows are constant-foldable by
# grappler. Defines module-level length and tolerance tables. No import-time side effects.
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
"""Tests for window_ops."""

import functools
import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.kernel_tests.signal import test_util
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.platform import test


_TF_DTYPE_TOLERANCE = [(dtypes.float16, 1e-2),
                       (dtypes.float32, 1e-6),
                       (dtypes.float64, 1e-9)]
_WINDOW_LENGTHS = [1, 2, 3, 4, 5, 31, 64, 128]
_MDCT_WINDOW_LENGTHS = [4, 16, 256]


def _scipy_raised_cosine(length, symmetric=True, a=0.5, b=0.5):
  # [FUNCTION DOC]
  # HELPER: NumPy reference window generator for the Hann and Hamming window tests.
  # Inputs: integer window length; symmetric flag (default true; when false a periodic window is
  # produced by building one extra sample for even lengths and dropping it); alpha and beta
  # cosine coefficients (default 0.5 each). Returns a float array: ones for length one,
  # otherwise alpha minus beta times a cosine over the sample index.
  """A simple implementation of a raised cosine window that matches SciPy.

  https://en.wikipedia.org/wiki/Window_function#Hann_window
  https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/windows.py#L615

  Args:
    length: The window length.
    symmetric: Whether to create a symmetric window.
    a: The alpha parameter of the raised cosine window.
    b: The beta parameter of the raised cosine window.

  Returns:
    A raised cosine window of length `length`.
  """
  if length == 1:
    return np.ones(1)
  odd = length % 2
  if not symmetric and not odd:
    length += 1
  window = a - b * np.cos(2.0 * np.pi * np.arange(length) / (length - 1))
  if not symmetric and not odd:
    window = window[:-1]
  return window


@tf_test_util.run_all_in_graph_and_eager_modes
class WindowOpsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for window_ops, run in graph and eager modes. Holds helper methods
  # to validate MDCT window properties and to compare TensorFlow windows to reference ones.

  def _check_mdct_window(self, window, tol=1e-6):
    # [FUNCTION DOC]
    # HELPER: shared assertions for the Vorbis and Kaiser-Bessel-derived window tests.
    # Takes an evaluated window array and a tolerance (default 1e-6). Asserts the length is a
    # multiple of four (Python assertion error otherwise), the Princen-Bradley condition (squares
    # of the two halves sum to one) and that the window is symmetric. Returns nothing.
    """Check that an MDCT window satisfies necessary conditions."""
    # We check that the length of the window is a multiple of 4 and
    # for symmetry of the window and also Princen-Bradley condition which
    # requires that  w[n]^2 + w[n + N//2]^2 = 1 for an N length window.
    wlen = int(np.shape(window)[0])
    assert wlen % 4 == 0
    half_len = wlen // 2
    squared_sums = window[:half_len]**2 + window[half_len:]**2
    self.assertAllClose(squared_sums, np.ones((half_len,)),
                        tol, tol)
    sym_diff = window[:half_len] - window[-1:half_len-1:-1]
    self.assertAllClose(sym_diff, np.zeros((half_len,)),
                        tol, tol)

  def _compare_window_fns(self, np_window_fn, tf_window_fn, window_length,
                          periodic, tf_dtype_tol):
    # [FUNCTION DOC]
    # HELPER: shared comparison for the Hann and Hamming window tests.
    # Takes a NumPy reference window function, a TensorFlow window function, a window length, a
    # periodic flag and a pair of TensorFlow dtype and tolerance. Builds the reference window
    # (symmetric when not periodic) cast to the dtype's NumPy type, builds the TensorFlow window
    # and asserts they are close within the tolerance. Returns nothing.
    tf_dtype, tol = tf_dtype_tol
    np_dtype = tf_dtype.as_numpy_dtype
    expected = np_window_fn(window_length,
                            symmetric=not periodic).astype(np_dtype)
    actual = tf_window_fn(window_length, periodic=periodic,
                          dtype=tf_dtype)
    self.assertAllClose(expected, actual, tol, tol)

  @parameterized.parameters(
      itertools.product(
          _WINDOW_LENGTHS,
          (4., 8., 10., 12.),
          _TF_DTYPE_TOLERANCE))
  def test_kaiser_window(self, window_length, beta, tf_dtype_tol):
    # [FUNCTION DOC]
    # Parameterized over window length, beta and dtype/tolerance. Asserts kaiser_window matches
    # the NumPy Kaiser window within the tolerance. Returns nothing.
    """Check that kaiser_window matches np.kaiser behavior."""
    self.assertAllClose(
        np.kaiser(window_length, beta),
        window_ops.kaiser_window(window_length, beta, tf_dtype_tol[0]),
        tf_dtype_tol[1], tf_dtype_tol[1])

  @parameterized.parameters(
      itertools.product(
          _WINDOW_LENGTHS,
          (False, True),
          _TF_DTYPE_TOLERANCE))
  def test_hann_window(self, window_length, periodic, tf_dtype_tol):
    # [FUNCTION DOC]
    # Parameterized over length, periodic flag and dtype/tolerance. Asserts hann_window matches
    # the raised cosine reference with alpha and beta of 0.5. Returns nothing.
    """Check that hann_window matches scipy.signal.hann behavior."""
    # The Hann window is a raised cosine window with parameters alpha=0.5 and
    # beta=0.5.
    # https://en.wikipedia.org/wiki/Window_function#Hann_window
    self._compare_window_fns(
        functools.partial(_scipy_raised_cosine, a=0.5, b=0.5),
        window_ops.hann_window, window_length, periodic, tf_dtype_tol)

  @parameterized.parameters(
      itertools.product(
          _WINDOW_LENGTHS,
          (False, True),
          _TF_DTYPE_TOLERANCE))
  def test_hamming_window(self, window_length, periodic, tf_dtype_tol):
    # [FUNCTION DOC]
    # Parameterized over length, periodic flag and dtype/tolerance. Asserts hamming_window
    # matches the raised cosine reference with alpha 0.54 and beta 0.46. Returns nothing.
    """Check that hamming_window matches scipy.signal.hamming's behavior."""
    # The Hamming window is a raised cosine window with parameters alpha=0.54
    # and beta=0.46.
    # https://en.wikipedia.org/wiki/Window_function#Hamming_window
    self._compare_window_fns(
        functools.partial(_scipy_raised_cosine, a=0.54, b=0.46),
        window_ops.hamming_window, window_length, periodic, tf_dtype_tol)

  @parameterized.parameters(
      itertools.product(
          (window_ops.hann_window, window_ops.hamming_window,
           window_ops.kaiser_window, window_ops.kaiser_bessel_derived_window,
           window_ops.vorbis_window),
          (False, True),
          _TF_DTYPE_TOLERANCE))
  def test_constant_folding(self, window_fn, periodic, tf_dtype_tol):
    # [FUNCTION DOC]
    # Parameterized over each window function, periodic flag and dtype/tolerance. Graph-mode
    # only. Builds a length-100 window in a fresh graph (retrying without the periodic argument
    # if the function rejects it with a type error), optimizes with grappler and asserts the
    # result has exactly one node. Returns nothing.
    """Window functions should be constant foldable for constant inputs."""
    if context.executing_eagerly():
      return
    g = ops.Graph()
    with g.as_default():
      try:
        window = window_fn(100, periodic=periodic, dtype=tf_dtype_tol[0])
      except TypeError:
        window = window_fn(100, dtype=tf_dtype_tol[0])
      rewritten_graph = test_util.grappler_optimize(g, [window])
      self.assertLen(rewritten_graph.node, 1)

  @parameterized.parameters(
      itertools.product(
          _MDCT_WINDOW_LENGTHS,
          _TF_DTYPE_TOLERANCE))
  def test_vorbis_window(self, window_length, tf_dtype_tol):
    # [FUNCTION DOC]
    # Parameterized over MDCT window length and dtype/tolerance. Builds a Vorbis window and runs
    # the MDCT window checks (Princen-Bradley and symmetry). Returns nothing.
    """Check if vorbis windows satisfy MDCT window conditions."""
    self._check_mdct_window(window_ops.vorbis_window(window_length,
                                                     dtype=tf_dtype_tol[0]),
                            tol=tf_dtype_tol[1])

  @parameterized.parameters(
      itertools.product(
          _MDCT_WINDOW_LENGTHS,
          (4., 8., 10., 12.),
          _TF_DTYPE_TOLERANCE))
  def test_kaiser_bessel_derived_window(self, window_length, beta,
                                        tf_dtype_tol):
    # [FUNCTION DOC]
    # Parameterized over MDCT window length, beta and dtype/tolerance. Builds a
    # Kaiser-Bessel-derived window and runs the MDCT window checks. Returns nothing.
    """Check if Kaiser-Bessel derived windows satisfy MDCT window conditions."""
    self._check_mdct_window(window_ops.kaiser_bessel_derived_window(
        window_length, beta=beta, dtype=tf_dtype_tol[0]), tol=tf_dtype_tol[1])

if __name__ == '__main__':
  test.main()
