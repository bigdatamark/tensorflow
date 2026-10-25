# [MODULE DOC]
# Test module checking that the special math ops (dawsn, expint, fresnel cosine and sine, spence,
# and the Bessel families i0, i0e, i1, i1e, j0, j1, k0, k0e, k1, k1e, y0, y1) accept WeakTensor
# inputs and return WeakTensor results with values matching SciPy where SciPy is available, plus
# gradient checks. Test classes are DawsnTest, ExpintTest, FresnelCosTest, FresnelSinTest,
# SpenceTest and BesselTest. Uses the weak tensor test utility helper to build inputs.
# Import-time side effect: importing the weak tensor ops module enables WeakTensor dispatch; when
# run as a script the dtype conversion mode is set to all before the test runner starts.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.ops.special_math_ops on WeakTensor."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import ops

from tensorflow.python.framework import test_util
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

_get_weak_tensor = weak_tensor_test_util.get_weak_tensor


@test_util.run_all_in_graph_and_eager_modes
class DawsnTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the Dawson integral op on WeakTensor inputs: boundary values, oddness, comparison
  # with SciPy for small and larger inputs, and gradient accuracy. Whole class runs in graph and
  # eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_dawsn_boundary(self):
    # [FUNCTION DOC]
    # Checks the Dawson integral of zero is zero and of not-a-number is not-a-number. Python
    # scalar inputs; no WeakTensor assertion.
    self.assertAllClose(0., special_math_ops.dawsn(0.))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.dawsn(np.nan))))

  @parameterized.parameters(np.float32, np.float64)
  def test_dawsn_odd(self, dtype):
    # [FUNCTION DOC]
    # For random float32/float64 weak tensors in [-100,100], checks the result is a WeakTensor and
    # that the function is odd (f(x) equals minus f(-x)). Parameterized by numpy dtype.
    x = _get_weak_tensor(
        np.random.uniform(-100.0, 100.0, size=int(1e4)).astype(dtype)
    )
    y = special_math_ops.dawsn(x)
    neg_y = -special_math_ops.dawsn(-x)

    self.assertIsInstance(y, WeakTensor)
    self.assertIsInstance(neg_y, WeakTensor)
    self.assertAllClose(self.evaluate(y), self.evaluate(neg_y))

  @parameterized.parameters(np.float32, np.float64)
  def test_dawsn_small(self, dtype):
    # [FUNCTION DOC]
    # Compares Dawson integral on 10000 random values in [-1,1] with SciPy and checks the result
    # is a WeakTensor. Parameterized by dtype. Skipped silently with a logged warning if SciPy
    # cannot be imported.
    x = np.random.uniform(-1., 1., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    y_wt = special_math_ops.dawsn(x_wt)
    self.assertIsInstance(y_wt, WeakTensor)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(special.dawsn(x), self.evaluate(y_wt))
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_dawsn_larger(self, dtype):
    # [FUNCTION DOC]
    # Compares Dawson integral on random values in [1,100] with SciPy and checks the result is a
    # WeakTensor. Parameterized by dtype. Skipped silently with a logged warning if SciPy cannot
    # be imported.
    x = np.random.uniform(1., 100., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    y_wt = special_math_ops.dawsn(x_wt)
    self.assertIsInstance(y_wt, WeakTensor)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(special.dawsn(x), y_wt)
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  def test_dawsn_gradient(self):
    # [FUNCTION DOC]
    # Compares analytical and numerical gradients of the Dawson integral on random weak inputs in
    # [-50,50]; the maximum error must be below 1e-4.
    inputs = [_get_weak_tensor(np.random.uniform(-50.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.dawsn, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)


@test_util.run_all_in_graph_and_eager_modes
class ExpintTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the exponential integral op on WeakTensor inputs: boundary values and domain,
  # comparison with SciPy for small and larger positive inputs, and gradient accuracy. Runs in
  # graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_expint_boundary(self):
    # [FUNCTION DOC]
    # Checks the exponential integral at zero is negative infinity, at not-a-number is
    # not-a-number, and is not-a-number for random negative inputs (domain is zero to infinity).
    self.assertAllClose(-np.inf, special_math_ops.expint(0.))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.expint(np.nan))))
    # Check that the domain of definition is [0, inf)
    self.assertTrue(
        np.all(
            np.isnan(
                self.evaluate(
                    special_math_ops.expint(
                        np.random.uniform(-20., -1., size=int(1e3)))))))

  @parameterized.parameters(np.float32, np.float64)
  def test_expint_small(self, dtype):
    # [FUNCTION DOC]
    # Compares the exponential integral on random values in [0,1] with SciPy expi and checks the
    # result is a WeakTensor. Parameterized by dtype. Skipped silently with a logged warning if
    # SciPy cannot be imported.
    x = np.random.uniform(0., 1., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    y_wt = special_math_ops.expint(x_wt)
    self.assertIsInstance(y_wt, WeakTensor)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.expi(x), self.evaluate(special_math_ops.expint(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_expint_larger(self, dtype):
    # [FUNCTION DOC]
    # Compares the exponential integral on random values in [1,50] with SciPy expi. Parameterized
    # by dtype. Does not assert the result is a WeakTensor. Skipped silently with a logged warning
    # if SciPy cannot be imported.
    x = np.random.uniform(1., 50., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.expi(x), self.evaluate(special_math_ops.expint(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  def test_expint_gradient(self):
    # [FUNCTION DOC]
    # Compares analytical and numerical gradients of the exponential integral on weak inputs in
    # [1,10]; maximum error must be below 5e-3.
    inputs = [_get_weak_tensor(np.random.uniform(1.0, 10.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.expint, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 5e-3)


@test_util.run_all_in_graph_and_eager_modes
class FresnelCosTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the Fresnel cosine integral op on WeakTensor inputs: boundaries, oddness, SciPy
  # comparison for small and larger inputs, and gradient accuracy. Runs in graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_fresnel_cos_boundary(self):
    # [FUNCTION DOC]
    # Checks the Fresnel cosine integral at zero is zero and at not-a-number is not-a-number.
    self.assertAllClose(0., special_math_ops.fresnel_cos(0.))
    self.assertTrue(
        np.isnan(self.evaluate(special_math_ops.fresnel_cos(np.nan))))

  @parameterized.parameters(np.float32, np.float64)
  def test_fresnel_cos_odd(self, dtype):
    # [FUNCTION DOC]
    # For random weak tensors in [-100,100], checks results are WeakTensors and the function is
    # odd. Parameterized by dtype.
    x = _get_weak_tensor(
        np.random.uniform(-100.0, 100.0, size=int(1e4)).astype(dtype)
    )
    y = special_math_ops.fresnel_cos(x)
    neg_y = -special_math_ops.fresnel_cos(-x)

    self.assertIsInstance(y, WeakTensor)
    self.assertIsInstance(neg_y, WeakTensor)
    self.assertAllClose(self.evaluate(y), self.evaluate(neg_y))

  @parameterized.parameters(np.float32, np.float64)
  def test_fresnel_cos_small(self, dtype):
    # [FUNCTION DOC]
    # Compares Fresnel cosine on random values in [0,1] with the second output of SciPy fresnel
    # and checks the result is a WeakTensor. Parameterized by dtype. Skipped silently with a
    # logged warning if SciPy cannot be imported.
    x = np.random.uniform(0., 1., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    y_wt = special_math_ops.fresnel_cos(x_wt)
    self.assertIsInstance(y_wt, WeakTensor)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(special.fresnel(x)[1], self.evaluate(y_wt))
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_fresnel_cos_larger(self, dtype):
    # [FUNCTION DOC]
    # Compares Fresnel cosine on random values in [1,100] with SciPy (relative tolerance 1e-5).
    # Parameterized by dtype. Does not assert a WeakTensor result. Skipped silently with a logged
    # warning if SciPy cannot be imported.
    x = np.random.uniform(1., 100., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.fresnel(x)[1],
          self.evaluate(special_math_ops.fresnel_cos(x_wt)),
          rtol=1e-5,
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  def test_fresnel_cos_gradient(self):
    # [FUNCTION DOC]
    # Compares analytical and numerical gradients of Fresnel cosine on weak inputs in [1,50];
    # maximum error must be below 5e-3.
    inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.fresnel_cos, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 5e-3)


@test_util.run_all_in_graph_and_eager_modes
class FresnelSinTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the Fresnel sine integral op on WeakTensor inputs: boundaries, oddness, SciPy
  # comparison for small and larger inputs, and gradient accuracy. Runs in graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_fresnel_sin_boundary(self):
    # [FUNCTION DOC]
    # Checks the Fresnel sine integral at zero is zero and at not-a-number is not-a-number.
    self.assertAllClose(0., special_math_ops.fresnel_sin(0.))
    self.assertTrue(
        np.isnan(self.evaluate(special_math_ops.fresnel_sin(np.nan))))

  @parameterized.parameters(np.float32, np.float64)
  def test_fresnel_sin_odd(self, dtype):
    # [FUNCTION DOC]
    # For random weak tensors in [-100,100], checks results are WeakTensors and the function is
    # odd, comparing the tensors directly without explicit evaluation. Parameterized by dtype.
    x = _get_weak_tensor(
        np.random.uniform(-100.0, 100.0, size=int(1e4)).astype(dtype)
    )
    y = special_math_ops.fresnel_sin(x)
    neg_y = -special_math_ops.fresnel_sin(-x)

    self.assertIsInstance(y, WeakTensor)
    self.assertIsInstance(neg_y, WeakTensor)
    self.assertAllClose(y, neg_y)

  @parameterized.parameters(np.float32, np.float64)
  def test_fresnel_sin_small(self, dtype):
    # [FUNCTION DOC]
    # Compares Fresnel sine on random values in [0,1] with the first output of SciPy fresnel.
    # Parameterized by dtype. Does not assert a WeakTensor result. Skipped silently with a logged
    # warning if SciPy cannot be imported.
    x = np.random.uniform(0., 1., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.fresnel(x)[0],
          self.evaluate(special_math_ops.fresnel_sin(x_wt)),
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_fresnel_sin_larger(self, dtype):
    # [FUNCTION DOC]
    # Compares Fresnel sine on random values in [1,100] with SciPy (relative tolerance 1e-5).
    # Parameterized by dtype. Skipped silently with a logged warning if SciPy cannot be imported.
    x = np.random.uniform(1., 100., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.fresnel(x)[0],
          self.evaluate(special_math_ops.fresnel_sin(x_wt)),
          rtol=1e-5,
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  def test_fresnel_sin_gradient(self):
    # [FUNCTION DOC]
    # Compares analytical and numerical gradients of Fresnel sine on weak inputs in [1,50];
    # maximum error must be below 5e-3.
    inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.fresnel_sin, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 5e-3)


@test_util.run_all_in_graph_and_eager_modes
class SpenceTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for Spence's dilogarithm op on WeakTensor inputs: boundary values and domain, SciPy
  # comparison for small and larger inputs, and gradients including at one. Runs in graph and
  # eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_spence_boundary(self):
    # [FUNCTION DOC]
    # Checks Spence's function equals pi squared over six at zero, zero at one, not-a-number at
    # not-a-number, and not-a-number for random negative inputs.
    self.assertAllClose(np.pi**2 / 6., special_math_ops.spence(0.))
    self.assertAllClose(0., special_math_ops.spence(1.))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.spence(np.nan))))
    # Check that the domain of definition is [0, inf)
    self.assertTrue(
        np.all(
            np.isnan(
                self.evaluate(
                    special_math_ops.spence(
                        np.random.uniform(-20., -1., size=int(1e3)))))))

  @parameterized.parameters(np.float32, np.float64)
  def test_spence_small(self, dtype):
    # [FUNCTION DOC]
    # Compares Spence's function on random values in [0,1] with SciPy and checks the result is a
    # WeakTensor. Parameterized by dtype. Skipped silently with a logged warning if SciPy cannot
    # be imported.
    x = np.random.uniform(0., 1., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    y_wt = special_math_ops.spence(x_wt)
    self.assertIsInstance(y_wt, WeakTensor)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(special.spence(x), self.evaluate(y_wt))
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_spence_larger(self, dtype):
    # [FUNCTION DOC]
    # Compares Spence's function on random values in [1,100] with SciPy. Parameterized by dtype.
    # Does not assert a WeakTensor result. Skipped silently with a logged warning if SciPy cannot
    # be imported.
    x = np.random.uniform(1., 100., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.spence(x), self.evaluate(special_math_ops.spence(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  def test_spence_gradient(self):
    # [FUNCTION DOC]
    # Compares analytical and numerical gradients of Spence's function on weak inputs in [1,50];
    # maximum error must be below 1e-4.
    inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.spence, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

  def test_spence_gradient_at_one(self):
    # [FUNCTION DOC]
    # Computes the analytical gradient of Spence's function at the plain Python value one and
    # checks it equals minus one (nested in the expected jacobian layout).
    analytical, _ = gradient_checker_v2.compute_gradient(
        special_math_ops.spence, [1.])
    self.assertAllClose([[[-1.]]], analytical)


@test_util.run_all_in_graph_and_eager_modes
class BesselTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the modified and ordinary Bessel function ops (i0, i0e, i1, i1e, j0, j1, k0, k0e,
  # k1, k1e, y0, y1) on WeakTensor inputs: boundaries, parity, SciPy comparisons for small and
  # larger inputs, and gradient accuracy. Runs in graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_besseli_boundary(self):
    # [FUNCTION DOC]
    # Checks Bessel i0 and i0e equal one at zero, i1 and i1e equal zero at zero, and all four
    # return not-a-number for not-a-number input.
    self.assertAllClose(1., special_math_ops.bessel_i0(0.))
    self.assertAllClose(1., special_math_ops.bessel_i0e(0.))
    self.assertAllClose(0., special_math_ops.bessel_i1(0.))
    self.assertAllClose(0., special_math_ops.bessel_i1e(0.))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_i0(np.nan))))
    self.assertTrue(
        np.isnan(self.evaluate(special_math_ops.bessel_i0e(np.nan))))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_i1(np.nan))))
    self.assertTrue(
        np.isnan(self.evaluate(special_math_ops.bessel_i1e(np.nan))))

  @test_util.run_in_graph_and_eager_modes
  def test_besselj_boundary(self):
    # [FUNCTION DOC]
    # Checks Bessel j0 equals one and j1 equals zero at zero, and both return not-a-number for
    # not-a-number input.
    self.assertAllClose(1., special_math_ops.bessel_j0(0.))
    self.assertAllClose(0., special_math_ops.bessel_j1(0.))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_j0(np.nan))))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_j1(np.nan))))

  @test_util.run_in_graph_and_eager_modes
  def test_besselk_boundary(self):
    # [FUNCTION DOC]
    # Checks Bessel k0, k0e, k1, k1e are infinite at zero and not-a-number for not-a-number input.
    self.assertTrue(np.isinf(self.evaluate(special_math_ops.bessel_k0(0.))))
    self.assertTrue(np.isinf(self.evaluate(special_math_ops.bessel_k0e(0.))))
    self.assertTrue(np.isinf(self.evaluate(special_math_ops.bessel_k1(0.))))
    self.assertTrue(np.isinf(self.evaluate(special_math_ops.bessel_k1e(0.))))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_k0(np.nan))))
    self.assertTrue(
        np.isnan(self.evaluate(special_math_ops.bessel_k0e(np.nan))))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_k1(np.nan))))
    self.assertTrue(
        np.isnan(self.evaluate(special_math_ops.bessel_k1e(np.nan))))

  @parameterized.parameters(np.float32, np.float64)
  def test_i0j0_even(self, dtype):
    # [FUNCTION DOC]
    # For random weak tensors in [-100,100], checks i0, i0e and j0 give the same values for x and
    # minus x (even functions). Parameterized by dtype.
    x = _get_weak_tensor(
        np.random.uniform(-100.0, 100.0, size=int(1e4)).astype(dtype)
    )
    self.assertAllClose(
        self.evaluate(special_math_ops.bessel_i0(x)),
        self.evaluate(special_math_ops.bessel_i0(-x)))

    self.assertAllClose(
        self.evaluate(special_math_ops.bessel_i0e(x)),
        self.evaluate(special_math_ops.bessel_i0e(-x)))

    self.assertAllClose(
        self.evaluate(special_math_ops.bessel_j0(x)),
        self.evaluate(special_math_ops.bessel_j0(-x)))

  @parameterized.parameters(np.float32, np.float64)
  def test_i1j1_odd(self, dtype):
    # [FUNCTION DOC]
    # For random weak tensors in [-100,100], checks i1, i1e and j1 are odd (value at x equals
    # minus value at minus x). Parameterized by dtype.
    x = _get_weak_tensor(
        np.random.uniform(-100.0, 100.0, size=int(1e4)).astype(dtype)
    )
    self.assertAllClose(
        self.evaluate(special_math_ops.bessel_i1(x)),
        self.evaluate(-special_math_ops.bessel_i1(-x)))

    self.assertAllClose(
        self.evaluate(special_math_ops.bessel_i1e(x)),
        self.evaluate(-special_math_ops.bessel_i1e(-x)))

    self.assertAllClose(
        self.evaluate(special_math_ops.bessel_j1(x)),
        self.evaluate(-special_math_ops.bessel_j1(-x)))

  @parameterized.parameters(np.float32, np.float64)
  def test_besseli_small(self, dtype):
    # [FUNCTION DOC]
    # Compares i0, i1, i0e, i1e on random values in [-1,1] with SciPy. Parameterized by dtype.
    # Skipped silently with a logged warning if SciPy cannot be imported.
    x = np.random.uniform(-1.0, 1.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.i0(x), self.evaluate(special_math_ops.bessel_i0(x_wt))
      )
      self.assertAllClose(
          special.i1(x), self.evaluate(special_math_ops.bessel_i1(x_wt))
      )
      self.assertAllClose(
          special.i0e(x), self.evaluate(special_math_ops.bessel_i0e(x_wt))
      )
      self.assertAllClose(
          special.i1e(x), self.evaluate(special_math_ops.bessel_i1e(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_besselj_small(self, dtype):
    # [FUNCTION DOC]
    # Compares j0 and j1 on random values in [-1,1] with SciPy. Parameterized by dtype. Skipped
    # silently with a logged warning if SciPy cannot be imported.
    x = np.random.uniform(-1.0, 1.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.j0(x), self.evaluate(special_math_ops.bessel_j0(x_wt))
      )
      self.assertAllClose(
          special.j1(x), self.evaluate(special_math_ops.bessel_j1(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_besselk_small(self, dtype):
    # [FUNCTION DOC]
    # Compares k0, k0e, k1, k1e on random values in [machine epsilon,1] with SciPy. Parameterized
    # by dtype. Skipped silently with a logged warning if SciPy cannot be imported.
    x = np.random.uniform(np.finfo(dtype).eps, 1.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.k0(x), self.evaluate(special_math_ops.bessel_k0(x_wt))
      )
      self.assertAllClose(
          special.k0e(x), self.evaluate(special_math_ops.bessel_k0e(x_wt))
      )
      self.assertAllClose(
          special.k1(x), self.evaluate(special_math_ops.bessel_k1(x_wt))
      )
      self.assertAllClose(
          special.k1e(x), self.evaluate(special_math_ops.bessel_k1e(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_bessely_small(self, dtype):
    # [FUNCTION DOC]
    # Compares y0 and y1 on random values in [machine epsilon,1] with SciPy. Parameterized by
    # dtype. Skipped silently with a logged warning if SciPy cannot be imported.
    x = np.random.uniform(np.finfo(dtype).eps, 1.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.y0(x), self.evaluate(special_math_ops.bessel_y0(x_wt))
      )
      self.assertAllClose(
          special.y1(x), self.evaluate(special_math_ops.bessel_y1(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_besseli_larger(self, dtype):
    # [FUNCTION DOC]
    # Compares i0e and i1e on random values in [1,20] with SciPy. Parameterized by dtype. Skipped
    # silently with a logged warning if SciPy cannot be imported.
    x = np.random.uniform(1.0, 20.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.i0e(x), self.evaluate(special_math_ops.bessel_i0e(x_wt))
      )
      self.assertAllClose(
          special.i1e(x), self.evaluate(special_math_ops.bessel_i1e(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_besselj_larger(self, dtype):
    # [FUNCTION DOC]
    # Compares j0 and j1 on random values in [1,30] with SciPy. Parameterized by dtype. Skipped
    # silently with a logged warning if SciPy cannot be imported.
    x = np.random.uniform(1.0, 30.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.j0(x), self.evaluate(special_math_ops.bessel_j0(x_wt))
      )
      self.assertAllClose(
          special.j1(x), self.evaluate(special_math_ops.bessel_j1(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_besselk_larger(self, dtype):
    # [FUNCTION DOC]
    # Compares k0, k0e, k1, k1e on random values in [1,30] with SciPy. Parameterized by dtype.
    # Skipped silently with a logged warning if SciPy cannot be imported.
    x = np.random.uniform(1.0, 30.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.k0(x), self.evaluate(special_math_ops.bessel_k0(x_wt))
      )
      self.assertAllClose(
          special.k0e(x), self.evaluate(special_math_ops.bessel_k0e(x_wt))
      )
      self.assertAllClose(
          special.k1(x), self.evaluate(special_math_ops.bessel_k1(x_wt))
      )
      self.assertAllClose(
          special.k1e(x), self.evaluate(special_math_ops.bessel_k1e(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_bessely_larger(self, dtype):
    # [FUNCTION DOC]
    # Compares y0 and y1 on random values in [1,30] with SciPy. Parameterized by dtype. Skipped
    # silently with a logged warning if SciPy cannot be imported.
    x = np.random.uniform(1.0, 30.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.y0(x), self.evaluate(special_math_ops.bessel_y0(x_wt))
      )
      self.assertAllClose(
          special.y1(x), self.evaluate(special_math_ops.bessel_y1(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  def test_besseli_gradient(self):
    # [FUNCTION DOC]
    # Checks numerical versus analytical gradients of i0, i0e, i1, i1e on weak inputs in [-10,10];
    # error limits 1e-3 for i0 and i1, 1e-4 for the scaled variants.
    inputs = [_get_weak_tensor(np.random.uniform(-10.0, 10.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_i0, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-3)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_i0e, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_i1, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-3)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_i1e, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

  def test_besselj_gradient(self):
    # [FUNCTION DOC]
    # Checks numerical versus analytical gradients of j0 and j1 on weak inputs in [-50,50];
    # maximum error below 1e-4.
    inputs = [_get_weak_tensor(np.random.uniform(-50.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_j0, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_j1, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

  def test_besselk_gradient(self):
    # [FUNCTION DOC]
    # Checks numerical versus analytical gradients of k0, k0e, k1, k1e on weak inputs in [1,50];
    # maximum error below 1e-4.
    inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_k0, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_k0e, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_k1, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_k1e, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

  def test_bessely_gradient(self):
    # [FUNCTION DOC]
    # Checks numerical versus analytical gradients of y0 and y1 on weak inputs in [1,50]; maximum
    # error below 1e-4.
    inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_y0, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_y1, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

# TODO(b/291943949): Add WeakTensor support for Einsum.

if __name__ == '__main__':
  ops.set_dtype_conversion_mode('all')
  test.main()
