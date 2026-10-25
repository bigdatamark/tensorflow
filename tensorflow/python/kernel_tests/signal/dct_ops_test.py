# [MODULE DOC]
# Tests for the discrete cosine transform ops (forward and inverse, types I to IV, with and
# without orthonormal scaling). Contains reference NumPy implementations of the four DCT types,
# a helper to pad or trim inputs, and a test class that compares the TensorFlow ops against the
# NumPy references and (when installed) SciPy, checks round trips, dynamic-shape tracing and
# error handling. Import-time side effect: tries to import SciPy's fftpack and logs a warning
# if it is missing.
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
"""Tests for DCT operations."""

import importlib
import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops.signal import dct_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def try_import(name):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: optional-dependency loader for the tests.
  # Input: name, dotted module name. Returns the imported module, or the null value if the
  # import fails, in which case a warning containing the error is logged. No exception escapes
  # for import errors.
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


fftpack = try_import("scipy.fftpack")


def _modify_input_for_dct(signals, n=None):
  # [FUNCTION DOC]
  # HELPER: input preparation for the NumPy reference DCT implementations.
  # KNOWN ISSUE: if n is given and is below one (zero or negative) no branch assigns the result
  # and the function fails with an unbound local variable error instead of a clear message.
  # Inputs: signals (array-like) and n (optional target length of the last axis; null means
  # keep as is). Returns the array unchanged when n is null or equal to the current length,
  # trimmed to the first n entries of the last axis when shorter, or zero-padded at the end
  # when longer. Asserts the resulting last-axis length equals n when n is truthy.
  """Pad or trim the provided NumPy array's innermost axis to length n."""
  signal = np.array(signals)
  if n is None or n == signal.shape[-1]:
    signal_mod = signal
  elif n >= 1:
    signal_len = signal.shape[-1]
    if n <= signal_len:
      signal_mod = signal[..., 0:n]
    else:
      output_shape = list(signal.shape)
      output_shape[-1] = n
      signal_mod = np.zeros(output_shape)
      signal_mod[..., 0:signal.shape[-1]] = signal
  if n:
    assert signal_mod.shape[-1] == n
  return signal_mod


def _np_dct1(signals, n=None, norm=None):
  # [FUNCTION DOC]
  # HELPER: NumPy reference for the type-I DCT used to check the TensorFlow op.
  # Inputs: signals (array), n (optional length to pad/trim the last axis to) and norm (ignored
  # and deleted since orthonormal scaling is not defined for type I). Returns the unnormalized
  # DCT-I along the last axis, computed with an explicit cosine sum per output index.
  """Computes the DCT-I manually with NumPy."""
  # X_k = (x_0 + (-1)**k * x_{N-1} +
  #       2 * sum_{n=0}^{N-2} x_n * cos(\frac{pi}{N-1} * n * k)  k=0,...,N-1
  del norm
  signals_mod = _modify_input_for_dct(signals, n=n)
  dct_size = signals_mod.shape[-1]
  dct = np.zeros_like(signals_mod)
  for k in range(dct_size):
    phi = np.cos(np.pi * np.arange(1, dct_size - 1) * k / (dct_size - 1))
    dct[..., k] = 2 * np.sum(
        signals_mod[..., 1:-1] * phi, axis=-1) + (
            signals_mod[..., 0] + (-1)**k * signals_mod[..., -1])
  return dct


def _np_dct2(signals, n=None, norm=None):
  # [FUNCTION DOC]
  # HELPER: NumPy reference for the type-II DCT used to check the TensorFlow op.
  # Inputs: signals (array), n (optional length to pad/trim the last axis to), norm (null or
  # the string for orthonormal scaling). Returns the DCT-II along the last axis using SciPy's
  # convention: a factor of two for no normalization, or orthonormal scaling when requested.
  """Computes the DCT-II manually with NumPy."""
  # X_k = sum_{n=0}^{N-1} x_n * cos(\frac{pi}{N} * (n + 0.5) * k)  k=0,...,N-1
  signals_mod = _modify_input_for_dct(signals, n=n)
  dct_size = signals_mod.shape[-1]
  dct = np.zeros_like(signals_mod)
  for k in range(dct_size):
    phi = np.cos(np.pi * (np.arange(dct_size) + 0.5) * k / dct_size)
    dct[..., k] = np.sum(signals_mod * phi, axis=-1)
  # SciPy's `dct` has a scaling factor of 2.0 which we follow.
  # https://github.com/scipy/scipy/blob/v1.2.1/scipy/fftpack/src/dct.c.src
  if norm == "ortho":
    # The orthonormal scaling includes a factor of 0.5 which we combine with
    # the overall scaling of 2.0 to cancel.
    dct[..., 0] *= np.sqrt(1.0 / dct_size)
    dct[..., 1:] *= np.sqrt(2.0 / dct_size)
  else:
    dct *= 2.0
  return dct


def _np_dct3(signals, n=None, norm=None):
  # [FUNCTION DOC]
  # HELPER: NumPy reference for the type-III DCT (inverse of type II) used to check the op.
  # Inputs: signals (array), n (optional length to pad/trim the last axis to), norm (null or
  # the string for orthonormal scaling). Works on a copy, scales it per SciPy's convention
  # (factor two, or orthonormal factors) and returns the DCT-III along the last axis.
  """Computes the DCT-III manually with NumPy."""
  # SciPy's `dct` has a scaling factor of 2.0 which we follow.
  # https://github.com/scipy/scipy/blob/v1.2.1/scipy/fftpack/src/dct.c.src
  signals_mod = _modify_input_for_dct(signals, n=n)
  dct_size = signals_mod.shape[-1]
  signals_mod = np.array(signals_mod)  # make a copy so we can modify
  if norm == "ortho":
    signals_mod[..., 0] *= np.sqrt(4.0 / dct_size)
    signals_mod[..., 1:] *= np.sqrt(2.0 / dct_size)
  else:
    signals_mod *= 2.0
  dct = np.zeros_like(signals_mod)
  # X_k = 0.5 * x_0 +
  #       sum_{n=1}^{N-1} x_n * cos(\frac{pi}{N} * n * (k + 0.5))  k=0,...,N-1
  half_x0 = 0.5 * signals_mod[..., 0]
  for k in range(dct_size):
    phi = np.cos(np.pi * np.arange(1, dct_size) * (k + 0.5) / dct_size)
    dct[..., k] = half_x0 + np.sum(signals_mod[..., 1:] * phi, axis=-1)
  return dct


def _np_dct4(signals, n=None, norm=None):
  # [FUNCTION DOC]
  # HELPER: NumPy reference for the type-IV DCT used to check the TensorFlow op.
  # Inputs: signals (array), n (optional length to pad/trim the last axis to), norm (null or
  # the string for orthonormal scaling). Works on a copy scaled by two or the orthonormal
  # factor, and returns the DCT-IV along the last axis.
  """Computes the DCT-IV manually with NumPy."""
  # SciPy's `dct` has a scaling factor of 2.0 which we follow.
  # https://github.com/scipy/scipy/blob/v1.2.1/scipy/fftpack/src/dct.c.src
  signals_mod = _modify_input_for_dct(signals, n=n)
  dct_size = signals_mod.shape[-1]
  signals_mod = np.array(signals_mod)  # make a copy so we can modify
  if norm == "ortho":
    signals_mod *= np.sqrt(2.0 / dct_size)
  else:
    signals_mod *= 2.0
  dct = np.zeros_like(signals_mod)
  # X_k = sum_{n=0}^{N-1}
  #            x_n * cos(\frac{pi}{4N} * (2n + 1) * (2k + 1))  k=0,...,N-1
  for k in range(dct_size):
    phi = np.cos(np.pi *
                 (2 * np.arange(0, dct_size) + 1) * (2 * k + 1) /
                 (4.0 * dct_size))
    dct[..., k] = np.sum(signals_mod * phi, axis=-1)
  return dct


NP_DCT = {1: _np_dct1, 2: _np_dct2, 3: _np_dct3, 4: _np_dct4}
NP_IDCT = {1: _np_dct1, 2: _np_dct3, 3: _np_dct2, 4: _np_dct4}


@test_util.run_all_in_graph_and_eager_modes
class DCTOpsTest(parameterized.TestCase, test.TestCase):
  # [CLASS DOC]
  # Parameterized tests of the DCT and inverse DCT ops, run in both graph and eager modes.
  # Uses the module-level NumPy reference implementations (and SciPy when available) as the
  # ground truth.

  def _compare(self, signals, n, norm, dct_type, atol, rtol):
    # [FUNCTION DOC]
    # HELPER: shared comparison used by the random test.
    # Inputs: signals (numpy array), n (optional DCT length), norm (null or orthonormal),
    # dct_type (1 to 4), atol and rtol (tolerances). Asserts that the TensorFlow forward and
    # inverse DCT keep the input dtype and match the NumPy reference, match SciPy for types
    # other than IV when SciPy is present, and that applying forward then inverse (and inverse
    # then forward) returns the input after undoing the scaling factor when norm is null.
    """Compares (I)DCT to SciPy (if available) and a NumPy implementation."""
    np_dct = NP_DCT[dct_type](signals, n=n, norm=norm)
    tf_dct = dct_ops.dct(signals, n=n, type=dct_type, norm=norm)
    self.assertEqual(tf_dct.dtype.as_numpy_dtype, signals.dtype)
    self.assertAllClose(np_dct, tf_dct, atol=atol, rtol=rtol)
    np_idct = NP_IDCT[dct_type](signals, n=None, norm=norm)
    tf_idct = dct_ops.idct(signals, type=dct_type, norm=norm)
    self.assertEqual(tf_idct.dtype.as_numpy_dtype, signals.dtype)
    self.assertAllClose(np_idct, tf_idct, atol=atol, rtol=rtol)
    if fftpack and dct_type != 4:
      scipy_dct = fftpack.dct(signals, n=n, type=dct_type, norm=norm)
      self.assertAllClose(scipy_dct, tf_dct, atol=atol, rtol=rtol)
      scipy_idct = fftpack.idct(signals, type=dct_type, norm=norm)
      self.assertAllClose(scipy_idct, tf_idct, atol=atol, rtol=rtol)
    # Verify inverse(forward(s)) == s, up to a normalization factor.
    # Since `n` is not implemented for IDCT operation, re-calculating tf_dct
    # without n.
    tf_dct = dct_ops.dct(signals, type=dct_type, norm=norm)
    tf_idct_dct = dct_ops.idct(tf_dct, type=dct_type, norm=norm)
    tf_dct_idct = dct_ops.dct(tf_idct, type=dct_type, norm=norm)
    if norm is None:
      if dct_type == 1:
        tf_idct_dct *= 0.5 / (signals.shape[-1] - 1)
        tf_dct_idct *= 0.5 / (signals.shape[-1] - 1)
      else:
        tf_idct_dct *= 0.5 / signals.shape[-1]
        tf_dct_idct *= 0.5 / signals.shape[-1]
    self.assertAllClose(signals, tf_idct_dct, atol=atol, rtol=rtol)
    self.assertAllClose(signals, tf_dct_idct, atol=atol, rtol=rtol)

  @parameterized.parameters(itertools.product(
      [1, 2, 3, 4],
      [None, "ortho"],
      [[2], [3], [10], [2, 20], [2, 3, 25]],
      [np.float32, np.float64]))
  def test_random(self, dct_type, norm, shape, dtype):
    # [FUNCTION DOC]
    # Parameterized over DCT type, normalization, input shape and float dtype. Skips (returns
    # early) the unsupported combination of type I with orthonormal scaling. Otherwise draws
    # random inputs and a random (or null) length, then runs the full comparison with tolerance
    # looser for single precision.
    """Test randomly generated batches of data."""
    # "ortho" normalization is not implemented for type I.
    if dct_type == 1 and norm == "ortho":
      return
    with self.session():
      tol = 5e-4 if dtype == np.float32 else 1e-7
      signals = np.random.rand(*shape).astype(dtype)
      n = np.random.randint(1, 2 * signals.shape[-1])
      n = np.random.choice([None, n])
      self._compare(signals, n, norm=norm, dct_type=dct_type,
                    rtol=tol, atol=tol)

  @parameterized.parameters(itertools.product(
      [1, 2, 3, 4],
      [None, "ortho"],
      [[2], [3], [10], [2, 20], [2, 3, 25]],
      [np.float32, np.float64]))
  def test_with_dynamic_dimensions(self, dct_type, norm, shape, dtype):
    # [FUNCTION DOC]
    # Parameterized like the random test. Skips type I with orthonormal scaling. Traces the DCT
    # in a compiled function using a spec with all dimensions unknown, then runs the concrete
    # function on real data; passes if tracing and execution raise nothing (values not checked).
    # "ortho" normalization is not implemented for type I.
    if dct_type == 1 and norm == "ortho":
      return
    signals = np.random.rand(*shape).astype(dtype)
    n = np.random.randint(1, 2 * shape[-1])
    n = np.random.choice([None, n])

    @def_function.function
    def func(signals):
      # [FUNCTION DOC]
      # Nested compiled function: applies the DCT of the enclosing test's type, length and
      # normalization to the given signals and returns the transformed tensor.
      return dct_ops.dct(signals, n=n, type=dct_type, norm=norm)

    # Trace with all undefined dimensions
    signals_spec = tensor_spec.TensorSpec([None] * len(shape), dtype)
    f = func.get_concrete_function(signals_spec)
    # Run with actual shape
    f(signals)

  def test_error(self):
    # [FUNCTION DOC]
    # Asserts the DCT op raises a value error for: unsupported type, negative length, type I
    # with orthonormal scaling, type I with a single element, and an unknown normalization
    # name; and raises a not-implemented error when a non-default axis is requested.
    signals = np.random.rand(10)
    # Unsupported type.
    with self.assertRaises(ValueError):
      dct_ops.dct(signals, type=5)
    # Invalid n.
    with self.assertRaises(ValueError):
      dct_ops.dct(signals, n=-2)
    # DCT-I normalization not implemented.
    with self.assertRaises(ValueError):
      dct_ops.dct(signals, type=1, norm="ortho")
    # DCT-I requires at least two inputs.
    with self.assertRaises(ValueError):
      dct_ops.dct(np.random.rand(1), type=1)
    # Unknown normalization.
    with self.assertRaises(ValueError):
      dct_ops.dct(signals, norm="bad")
    with self.assertRaises(NotImplementedError):
      dct_ops.dct(signals, axis=0)


if __name__ == "__main__":
  test.main()
