# [MODULE DOC]
# Unit tests for the special math functions in the distributions ops package: the normal
# quantile (ndtri), normal CDF (ndtr), log normal CDF (log_ndtr), inverse error function
# (erfinv) and log CDF of the Laplace distribution (log_cdf_laplace). Results are compared
# against scipy.special / scipy.stats when scipy can be imported (otherwise most tests return
# early without checking anything), and gradients are checked for finiteness and positivity.
# Contains small module-level helpers (optional-import, grid generation, value-and-gradient)
# plus test classes. Import-time side effects: tries to import scipy and logs a warning if
# unavailable. Run as a test binary via the TensorFlow test main entry point.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Special Math Ops."""

import collections
import importlib

import numpy as np

from tensorflow.python.eager import backprop as tfe_backprop
from tensorflow.python.eager import context as tfe_context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops.distributions import special_math
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def try_import(name):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: optional-dependency loader used to import scipy modules for the reference checks.
  # Imports the module with the given dotted name and returns it. If the import fails with an
  # import error, logs a warning (side effect) containing the module name and the error text and
  # returns the null value instead of raising. Other exceptions propagate.
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


special = try_import("scipy.special")
stats = try_import("scipy.stats")
sm = special_math


def _check_strictly_increasing(array_1d):
  # [FUNCTION DOC]
  # HELPER: assertion utility shared by the grid tests in this module.
  # Takes a one-dimensional numeric array, computes successive differences and asserts that all
  # of them are strictly greater than zero, i.e. the sequence is strictly increasing. Raises an
  # assertion error otherwise. Returns nothing.
  diff = np.diff(array_1d)
  np.testing.assert_array_less(0, diff)


def _make_grid(dtype, grid_spec):
  # [FUNCTION DOC]
  # HELPER: test-input generator for the grid-based accuracy tests.
  # Inputs: dtype is the numpy element type of the result; grid_spec is a record with the lower
  # bound, upper bound and target shape. Builds evenly spaced points between the bounds (one per
  # element of the shape), adds small seeded random noise (fixed seed, so deterministic; noise is
  # a tenth of the grid spacing), sorts the points ascending and reshapes to the requested shape.
  # Returns the numpy array. No side effects.
  """Returns a uniform grid + noise, reshaped to shape argument."""
  rng = np.random.RandomState(0)
  num_points = np.prod(grid_spec.shape)
  grid = np.linspace(grid_spec.min, grid_spec.max, num=num_points).astype(dtype)
  grid_spacing = (grid_spec.max - grid_spec.min) / num_points
  grid += 0.1 * grid_spacing * rng.randn(*grid.shape)  # pylint: disable=not-an-iterable
  # More useful if it's sorted (e.g. for testing monotonicity, or debugging).
  grid = np.sort(grid)
  return np.reshape(grid, grid_spec.shape)


def _value_and_gradient(fn, *args):
  # [FUNCTION DOC]
  # HELPER: computes a function value and its gradient in either eager or graph mode.
  # KNOWN ISSUE: the docstring says the gradient is taken with respect to a single `arg`, but the
  # function accepts any number of positional arguments and returns the gradient with respect to
  # all of them as a sequence (a list/tuple with one entry per argument).
  # Inputs: fn is a callable on tensors; args are the tensors to differentiate against. When
  # executing eagerly, uses the eager backprop value-and-gradient helper on the argument tuple;
  # otherwise calls fn directly and builds symbolic gradients of the result. Returns a pair of
  # the value and the gradients.
  """Calls `fn` and computes the gradient of the result wrt `arg`."""
  if tfe_context.executing_eagerly():
    v, g = tfe_backprop.val_and_grad_function(fn)(args)
  else:
    v = fn(*args)
    g = gradients_impl.gradients(v, args)
  return v, g


GridSpec = collections.namedtuple("GridSpec", ["min", "max", "shape"])

ErrorSpec = collections.namedtuple("ErrorSpec", ["rtol", "atol"])


class NdtriTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the normal quantile function (ndtri, inverse normal CDF): accuracy against scipy,
  # behavior with unknown static shape, and finiteness of gradients at the boundary points of
  # its piecewise approximation. Plain test case with no shared fixtures beyond a finiteness
  # assertion helper.

  def assertAllFinite(self, x):
    # [FUNCTION DOC]
    # HELPER: custom assertion for this test class.
    # Asserts that every element of the given array is finite (not NaN and not infinite) by
    # comparing the finiteness mask against an all-true array. Raises an assertion error on
    # failure; returns nothing.
    is_finite = np.isfinite(x)
    all_true = np.ones_like(is_finite, dtype=np.bool_)
    self.assertAllEqual(all_true, is_finite)

  @test_util.run_in_graph_and_eager_modes
  def testNdtri(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): evaluates ndtri on 50 evenly spaced probabilities in the
    # unit interval plus extra values near zero, near one and at two interior breakpoints so
    # every piece of the piecewise approximation is hit, and asserts the result equals scipy's
    # ndtri with zero absolute tolerance (default relative tolerance). Returns early, asserting
    # nothing, when scipy is unavailable.
    """Verifies that ndtri computation is correct."""
    if not special:
      return

    p = np.linspace(0., 1.0, 50).astype(np.float64)
    # Quantile performs piecewise rational approximation so adding some
    # special input values to make sure we hit all the pieces.
    p = np.hstack((p, np.exp(-32), 1. - np.exp(-32), np.exp(-2),
                   1. - np.exp(-2)))
    expected_x = special.ndtri(p)
    x = special_math.ndtri(p)
    self.assertAllClose(expected_x, self.evaluate(x), atol=0.)

  @test_util.run_deprecated_v1
  def testNdtriDynamicShape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is a copy of the one for testNdtri and does not mention what
    # is distinctive here, namely that the input is a float32 placeholder with unspecified
    # shape fed at run time.
    # Test (deprecated graph mode only): builds ndtri on a float32 placeholder, feeds 50 evenly
    # spaced probabilities through a session and asserts equality with scipy's ndtri (zero
    # absolute tolerance). Returns early without asserting when scipy is unavailable.
    """Verifies that ndtri computation is correct."""
    with self.cached_session() as sess:
      if not special:
        return

      p = array_ops.placeholder(np.float32)
      p_ = np.linspace(0., 1.0, 50).astype(np.float32)

      x = special_math.ndtri(p)
      x_ = sess.run(x, feed_dict={p: p_})

      expected_x_ = special.ndtri(p_)
      self.assertAllClose(expected_x_, x_, atol=0.)

  def _baseNdtriFiniteGradientTest(self, dtype):
    # [FUNCTION DOC]
    # HELPER: shared body of the float32 and float64 finite-gradient tests.
    # Input: dtype, the numpy float type to use. Builds a constant of probabilities at zero, one
    # and the approximation's piece boundaries (exp(-32), exp(-2) and their complements),
    # computes ndtri and its gradient (through a wrapping function so extra default arguments
    # cannot break the gradient helper) and asserts the gradient values are all finite.
    """Verifies that ndtri has finite gradients at interesting points."""
    # Tests gradients at 0, 1, and piece-wise boundaries.
    p = constant_op.constant(
        np.array([
            0.,
            np.exp(-32.),
            np.exp(-2.),
            1. - np.exp(-2.),
            1. - np.exp(-32.),
            1.,
        ]).astype(dtype))
    # Not having the lambda sanitizer means we'd get an `IndexError` whenever
    # the user supplied function has default args.
    _, grads = _value_and_gradient(
        lambda x: special_math.ndtri(x), p)  # pylint: disable=unnecessary-lambda
    self.assertAllFinite(self.evaluate(grads[0]))

  @test_util.run_in_graph_and_eager_modes
  def testNdtriFiniteGradientFloat32(self):
    # [FUNCTION DOC]
    # Test (graph and eager): runs the shared finite-gradient check with single precision.
    self._baseNdtriFiniteGradientTest(np.float32)

  @test_util.run_in_graph_and_eager_modes
  def testNdtriFiniteGradientFloat64(self):
    # [FUNCTION DOC]
    # Test (graph and eager): runs the shared finite-gradient check with double precision.
    self._baseNdtriFiniteGradientTest(np.float64)


@test_util.run_all_in_graph_and_eager_modes
class NdtrTest(test.TestCase):
  # [CLASS DOC]
  # Tests the normal CDF (ndtr) over a grid of inputs in float32 and float64, in both graph and
  # eager modes (class-level decorator). Class attributes configure the run: a flag selecting
  # whether the log CDF is tested instead, the float32/float64 grid specifications (range and
  # shape) chosen so the CDF stays strictly between 0 and 1, and the relative/absolute error
  # tolerances. Subclasses override these attributes to cover the log CDF in different input
  # ranges.
  _use_log = False
  # Grid min/max chosen to ensure 0 < cdf(x) < 1.
  _grid32 = GridSpec(min=-12.9, max=5., shape=[100])
  _grid64 = GridSpec(min=-37.5, max=8., shape=[100])
  _error32 = ErrorSpec(rtol=1e-4, atol=0.)
  _error64 = ErrorSpec(rtol=1e-6, atol=0.)

  def _test_grid(self, dtype, grid_spec, error_spec):
    # [FUNCTION DOC]
    # HELPER: dispatcher for the grid tests. Inputs: numpy dtype, grid specification and error
    # specification (relative and absolute tolerances). Calls the log-CDF variant when the
    # class flag for logs is set, otherwise the plain-CDF variant. Returns nothing.
    if self._use_log:
      self._test_grid_log(dtype, grid_spec, error_spec)
    else:
      self._test_grid_no_log(dtype, grid_spec, error_spec)

  def _test_grid_log(self, dtype, grid_spec, error_spec):
    # [FUNCTION DOC]
    # HELPER: grid check for log_ndtr, used by the log-CDF subclasses via the dispatcher.
    # Inputs: numpy dtype, grid specification, error specification. Returns early when scipy is
    # missing. Otherwise builds the grid, evaluates log_ndtr, asserts every value is finite,
    # negative and strictly increasing, then compares with scipy's log_ndtr using the given
    # tolerances, restricted to points where scipy's result is negative (scipy underflows to
    # zero in places where this implementation does not).
    if not special:
      return

    grid = _make_grid(dtype, grid_spec)
    actual = self.evaluate(sm.log_ndtr(grid))

    # Basic tests.
    # isfinite checks for NaN and Inf.
    self.assertTrue(np.isfinite(actual).all())
    # On the grid, -inf < log_cdf(x) < 0.  In this case, we should be able
    # to use a huge grid because we have used tricks to escape numerical
    # difficulties.
    self.assertTrue((actual < 0).all())
    _check_strictly_increasing(actual)

    # Versus scipy.
    expected = special.log_ndtr(grid)
    # Scipy prematurely goes to zero at some places that we don't.  So don't
    # include these in the comparison.
    self.assertAllClose(
        expected.astype(np.float64)[expected < 0],
        actual.astype(np.float64)[expected < 0],
        rtol=error_spec.rtol,
        atol=error_spec.atol)

  def _test_grid_no_log(self, dtype, grid_spec, error_spec):
    # [FUNCTION DOC]
    # HELPER: grid check for ndtr, used by the dispatcher when the log flag is unset.
    # KNOWN ISSUE: the final comparison with scipy selects only points where scipy's ndtr is
    # negative. A CDF is never negative, so the selection is empty and that assertion compares
    # nothing (vacuous); only the finite, strictly-between-0-and-1 and strictly-increasing checks
    # are effective. It was probably copied from the log variant.
    # Inputs: numpy dtype, grid specification, error specification. Returns early without
    # asserting when scipy is missing.
    if not special:
      return

    grid = _make_grid(dtype, grid_spec)
    actual = self.evaluate(sm.ndtr(grid))

    # Basic tests.
    # isfinite checks for NaN and Inf.
    self.assertTrue(np.isfinite(actual).all())
    # On the grid, 0 < cdf(x) < 1.  The grid cannot contain everything due
    # to numerical limitations of cdf.
    self.assertTrue((actual > 0).all())
    self.assertTrue((actual < 1).all())
    _check_strictly_increasing(actual)

    # Versus scipy.
    expected = special.ndtr(grid)
    # Scipy prematurely goes to zero at some places that we don't.  So don't
    # include these in the comparison.
    self.assertAllClose(
        expected.astype(np.float64)[expected < 0],
        actual.astype(np.float64)[expected < 0],
        rtol=error_spec.rtol,
        atol=error_spec.atol)

  @test_util.run_deprecated_v1
  def test_float32(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): runs the grid check in float32 using the class's float32
    # grid and tolerance settings. Inherited by the log-CDF subclasses.
    self._test_grid(np.float32, self._grid32, self._error32)

  @test_util.run_deprecated_v1
  def test_float64(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): runs the grid check in float64 using the class's float64
    # grid and tolerance settings. Inherited by the log-CDF subclasses.
    self._test_grid(np.float64, self._grid64, self._error64)


class LogNdtrTestLower(NdtrTest):
  # [CLASS DOC]
  # Runs the inherited grid tests on log_ndtr for the lower segment: inputs from -100 up to the
  # float-specific lower threshold below which a asymptotic series is used. Overrides the log
  # flag, grids and tolerances only.
  _use_log = True
  _grid32 = GridSpec(min=-100., max=sm.LOGNDTR_FLOAT32_LOWER, shape=[100])
  _grid64 = GridSpec(min=-100., max=sm.LOGNDTR_FLOAT64_LOWER, shape=[100])
  _error32 = ErrorSpec(rtol=1e-4, atol=0.)
  _error64 = ErrorSpec(rtol=1e-4, atol=0.)


# The errors are quite large when the input is > 6 or so.  Also,
# scipy.special.log_ndtr becomes zero very early, before 10,
# (due to ndtr becoming 1).  We approximate Log[1 + epsilon] as epsilon, and
# avoid this issue.
class LogNdtrTestMid(NdtrTest):
  # [CLASS DOC]
  # Runs the inherited grid tests on log_ndtr for the middle segment, between the lower and
  # upper float-specific thresholds, with loose relative tolerance (0.1) plus a small absolute
  # tolerance because the tail differs from scipy. Overrides the log flag, grids and tolerances.
  _use_log = True
  _grid32 = GridSpec(
      min=sm.LOGNDTR_FLOAT32_LOWER, max=sm.LOGNDTR_FLOAT32_UPPER, shape=[100])
  _grid64 = GridSpec(
      min=sm.LOGNDTR_FLOAT64_LOWER, max=sm.LOGNDTR_FLOAT64_UPPER, shape=[100])
  # Differences show up as soon as we're in the tail, so add some atol.
  _error32 = ErrorSpec(rtol=0.1, atol=1e-7)
  _error64 = ErrorSpec(rtol=0.1, atol=1e-7)


class LogNdtrTestUpper(NdtrTest):
  # [CLASS DOC]
  # Runs the inherited grid tests on log_ndtr for the upper segment, from the upper threshold
  # to 12 (float32) or 35 (float64), beyond which the log CDF may be zero. Tight relative
  # tolerance with a tiny absolute tolerance. Overrides the log flag, grids and tolerances.
  _use_log = True
  _grid32 = GridSpec(
      min=sm.LOGNDTR_FLOAT32_UPPER,
      max=12.,  # Beyond this, log_cdf(x) may be zero.
      shape=[100])
  _grid64 = GridSpec(
      min=sm.LOGNDTR_FLOAT64_UPPER,
      max=35.,  # Beyond this, log_cdf(x) may be zero.
      shape=[100])
  _error32 = ErrorSpec(rtol=1e-6, atol=1e-14)
  _error64 = ErrorSpec(rtol=1e-6, atol=1e-14)


class NdtrGradientTest(test.TestCase):
  # [CLASS DOC]
  # Tests gradients of the normal CDF (ndtr) over a wide grid (-100 to 100, four-dimensional
  # shape) in float32 and float64: finiteness at selected points, positivity/non-negativity,
  # absence of NaN, and a comparison with the normal density from scipy. Class attributes set
  # the log flag, grid and tolerances; the log subclass flips the flag to test log_ndtr.
  # Provides small boolean-array assertion helpers.
  _use_log = False
  _grid = GridSpec(min=-100., max=100., shape=[1, 2, 3, 8])
  _error32 = ErrorSpec(rtol=1e-4, atol=0)
  _error64 = ErrorSpec(rtol=1e-7, atol=0)

  def assert_all_true(self, v):
    # [FUNCTION DOC]
    # HELPER: custom assertion. Asserts every element of the boolean array v is true; raises an
    # assertion error otherwise. Returns nothing.
    self.assertAllEqual(np.ones_like(v, dtype=np.bool_), v)

  def assert_all_false(self, v):
    # [FUNCTION DOC]
    # HELPER: custom assertion. Asserts every element of the boolean array v is false; raises an
    # assertion error otherwise. Returns nothing.
    self.assertAllEqual(np.zeros_like(v, dtype=np.bool_), v)

  def _test_grad_finite(self, dtype):
    # [FUNCTION DOC]
    # HELPER: shared by the dtype tests. Input: numpy dtype. Evaluates ndtr (or log_ndtr when
    # the log flag is set) and its gradient at the three points -100, 0 and 100 and asserts
    # that both the values and the gradient are finite. The function is wrapped in a lambda so
    # the gradient helper does not choke on default arguments.
    x = constant_op.constant([-100., 0., 100.], dtype=dtype)
    output = (sm.log_ndtr(x) if self._use_log else sm.ndtr(x))
    fn = sm.log_ndtr if self._use_log else sm.ndtr
    # Not having the lambda sanitizer means we'd get an `IndexError` whenever
    # the user supplied function has default args.
    output, grad_output = _value_and_gradient(
        lambda x_: fn(x_), x)  # pylint: disable=unnecessary-lambda
    # isfinite checks for NaN and Inf.
    output_, grad_output_ = self.evaluate([output, grad_output])
    self.assert_all_true(np.isfinite(output_))
    self.assert_all_true(np.isfinite(grad_output_[0]))

  def _test_grad_accuracy(self, dtype, grid_spec, error_spec):
    # [FUNCTION DOC]
    # HELPER: shared by the dtype tests; needs a graph-mode session.
    # KNOWN ISSUE: the final scipy comparison selects points where the expected derivative is
    # negative; the normal density (and its ratio to the CDF) is never negative, so the
    # selection is empty and the comparison is vacuous. Only the sign, NaN and finiteness
    # checks are effective. Also the "expected" array may be modified in place for the log case.
    # Inputs: numpy dtype, grid specification, error specification. Builds the grid, evaluates
    # the numerical Jacobian of the function via the gradient checker and checks its diagonal
    # is non-NaN, positive and finite; then evaluates the symbolic gradient and checks non-NaN,
    # positive (first half) / non-negative (rest) for the log case, non-negative for the plain
    # case, and finite. Returns early before the scipy comparison if scipy is unavailable.
    raw_grid = _make_grid(dtype, grid_spec)
    grid = ops.convert_to_tensor(raw_grid)
    with self.cached_session():
      fn = sm.log_ndtr if self._use_log else sm.ndtr

      # If there are N points in the grid,
      # grad_eval.shape = (N, N), with grad_eval[i, j] the partial derivative of
      # the ith output point w.r.t. the jth grid point.  We only expect the
      # diagonal to be nonzero.
      # TODO(b/31131137): Replace tf.compat.v1.test.compute_gradient with our
      # own custom gradient evaluation to ensure we correctly handle small
      # function delta.
      grad_eval, _ = gradient_checker.compute_gradient(grid, grid_spec.shape,
                                                       fn(grid),
                                                       grid_spec.shape)
      grad_eval = np.diag(grad_eval)

      # Check for NaN separately in order to get informative failures.
      self.assert_all_false(np.isnan(grad_eval))
      self.assert_all_true(grad_eval > 0.)
      # isfinite checks for NaN and Inf.
      self.assert_all_true(np.isfinite(grad_eval))

      # Do the same checks but explicitly compute the gradient.
      # (We did this because we're not sure if we trust
      # tf.test.compute_gradient.)
      grad_eval = gradients_impl.gradients(fn(grid), grid)[0].eval()
      self.assert_all_false(np.isnan(grad_eval))
      if self._use_log:
        g = np.reshape(grad_eval, [-1])
        half = np.ceil(len(g) / 2)
        self.assert_all_true(g[:int(half)] > 0.)
        self.assert_all_true(g[int(half):] >= 0.)
      else:
        # The ndtr gradient will only be non-zero in the range [-14, 14] for
        # float32 and [-38, 38] for float64.
        self.assert_all_true(grad_eval >= 0.)
      # isfinite checks for NaN and Inf.
      self.assert_all_true(np.isfinite(grad_eval))

      # Versus scipy.
      if not (special and stats):
        return

      expected = stats.norm.pdf(raw_grid)
      if self._use_log:
        expected /= special.ndtr(raw_grid)
        expected[np.isnan(expected)] = 0.
      # Scipy prematurely goes to zero at some places that we don't.  So don't
      # include these in the comparison.
      self.assertAllClose(
          expected.astype(np.float64)[expected < 0],
          grad_eval.astype(np.float64)[expected < 0],
          rtol=error_spec.rtol,
          atol=error_spec.atol)

  @test_util.run_deprecated_v1
  def test_float32(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): runs the gradient accuracy check and the finite-gradient
    # check in single precision using the class's grid and float32 tolerances.
    self._test_grad_accuracy(np.float32, self._grid, self._error32)
    self._test_grad_finite(np.float32)

  @test_util.run_deprecated_v1
  def test_float64(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): runs the gradient accuracy check and the finite-gradient
    # check in double precision using the class's grid and float64 tolerances.
    self._test_grad_accuracy(np.float64, self._grid, self._error64)
    self._test_grad_finite(np.float64)


class LogNdtrGradientTest(NdtrGradientTest):
  # [CLASS DOC]
  # Repeats all gradient tests of the parent class for log_ndtr by setting the log flag; no
  # other overrides.
  _use_log = True


class ErfInvTest(test.TestCase):
  # [CLASS DOC]
  # Tests the inverse error function: numeric agreement with scipy and rejection of integer
  # input. Plain test case without shared fixtures.

  def testErfInvValues(self):
    # [FUNCTION DOC]
    # Test: evaluates erfinv on 50 evenly spaced float64 values from 0 to 1 (inclusive, so the
    # last value maps to infinity in both implementations) and asserts equality with scipy's
    # erfinv with zero absolute tolerance. Returns early when scipy is unavailable.
    with self.cached_session():
      if not special:
        return

      x = np.linspace(0., 1.0, 50).astype(np.float64)

      expected_x = special.erfinv(x)
      x = special_math.erfinv(x)
      self.assertAllClose(expected_x, self.evaluate(x), atol=0.)

  def testErfInvIntegerInput(self):
    # [FUNCTION DOC]
    # Test: asserts that erfinv raises a type error when given a 32-bit integer array and
    # again when given a 64-bit integer array, since only floating point input is supported.
    with self.cached_session():

      with self.assertRaises(TypeError):
        x = np.array([1, 2, 3]).astype(np.int32)
        special_math.erfinv(x)

      with self.assertRaises(TypeError):
        x = np.array([1, 2, 3]).astype(np.int64)
        special_math.erfinv(x)


class LogCDFLaplaceTest(test.TestCase):
  # [CLASS DOC]
  # Tests log_cdf_laplace, the log CDF of a standard Laplace distribution, over grids in
  # float32 and float64: finiteness, negativity, monotonicity, agreement with scipy where scipy
  # is numerically stable, and finite non-zero values and gradients in extreme ranges. Class
  # constants give the largest inputs for which the naive implementation still works in each
  # precision. Provides an all-true assertion helper.
  # Note that scipy.stats.laplace does not have a stable Log CDF, so we cannot
  # rely on scipy to cross check the extreme values.

  # Test will be done differently over different ranges.  These are the values
  # such that when exceeded by x, produce output that causes the naive (scipy)
  # implementation to have numerical issues.
  #
  # If x = log(1 / (2 * eps)), then 0.5 * exp{-x} = eps.
  # With inserting eps = np.finfo(dtype).eps, we see that log(1 / (2 * eps)) is
  # the value of x such that any larger value will result in
  # 1 - 0.5 * exp{-x} = 0, which will cause the log_cdf_laplace code to take a
  # log # of zero.  We therefore choose these as our cutoffs for testing.
  CUTOFF_FLOAT64_UPPER = np.log(1. / (2. * np.finfo(np.float64).eps)) - 1.
  CUTOFF_FLOAT32_UPPER = np.log(1. / (2. * np.finfo(np.float32).eps)) - 1.

  def assertAllTrue(self, x):
    # [FUNCTION DOC]
    # HELPER: custom assertion. Asserts every element of the boolean array x is true; raises an
    # assertion error otherwise. Returns nothing.
    self.assertAllEqual(np.ones_like(x, dtype=np.bool_), x)

  def _test_grid_log(self, dtype, scipy_dtype, grid_spec, error_spec):
    # [FUNCTION DOC]
    # HELPER: shared body for the grid-based tests below.
    # Inputs: dtype for the grid passed to the function under test, scipy_dtype for the grid
    # passed to scipy, grid specification, error specification. Evaluates log_cdf_laplace in a
    # session, asserts values are finite, negative and strictly increasing, then (when scipy
    # stats is available) compares against scipy's standard Laplace log CDF with the given
    # tolerances in float64.
    with self.cached_session():
      grid = _make_grid(dtype, grid_spec)
      actual = sm.log_cdf_laplace(grid).eval()

      # Basic tests.
      # isfinite checks for NaN and Inf.
      self.assertAllTrue(np.isfinite(actual))
      self.assertAllTrue((actual < 0))
      _check_strictly_increasing(actual)

      # Versus scipy.
      if not stats:
        return

      scipy_dist = stats.laplace(loc=0., scale=1.)
      expected = scipy_dist.logcdf(grid.astype(scipy_dtype))
      self.assertAllClose(
          expected.astype(np.float64),
          actual.astype(np.float64),
          rtol=error_spec.rtol,
          atol=error_spec.atol)

  @test_util.run_deprecated_v1
  def test_float32_lower_and_mid_segment_scipy_float32_ok(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): checks float32 inputs from -10 to just below the float32
    # cutoff (minus 5) against scipy computed in float32 with relative tolerance 5e-4.
    # Choose values mild enough that we can use scipy in float32, which will
    # allow for a high accuracy match to scipy (since we both use float32).
    self._test_grid_log(
        np.float32,  # dtype
        np.float32,  # scipy_dtype
        GridSpec(min=-10, max=self.CUTOFF_FLOAT32_UPPER - 5, shape=[100]),
        ErrorSpec(rtol=5e-4, atol=0))

  @test_util.run_deprecated_v1
  def test_float32_all_segments_with_scipy_float64_ok(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): checks float32 inputs from -50 to 5 beyond the float32
    # cutoff, covering all segments, against scipy computed in float64, with a loose relative
    # tolerance of 0.05 because this side runs in single precision.
    # Choose values outside the range where scipy float32 works.
    # Let scipy use float64.  This means we
    # won't be exactly the same since we are in float32.
    self._test_grid_log(
        np.float32,  # dtype
        np.float64,  # scipy_dtype
        GridSpec(min=-50, max=self.CUTOFF_FLOAT32_UPPER + 5, shape=[100]),
        ErrorSpec(rtol=0.05, atol=0))

  @test_util.run_deprecated_v1
  def test_float32_extreme_values_result_and_gradient_finite_and_nonzero(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): on a 20 by 100 float32 grid from -200 to 80, evaluates
    # log_cdf_laplace and its gradient and asserts all values and gradients are finite and none
    # is exactly zero. The session variable is created but unused.
    with self.cached_session() as sess:
      # On the lower branch, log_cdf_laplace(x) = x, so we know this will be
      # fine, but test to -200 anyways.
      grid = _make_grid(
          np.float32, GridSpec(min=-200, max=80, shape=[20, 100]))
      grid = ops.convert_to_tensor(grid)

      actual = sm.log_cdf_laplace(grid)
      grad = gradients_impl.gradients(actual, grid)[0]

      actual_, grad_ = self.evaluate([actual, grad])

      # isfinite checks for NaN and Inf.
      self.assertAllTrue(np.isfinite(actual_))
      self.assertAllTrue(np.isfinite(grad_))
      self.assertFalse(np.any(actual_ == 0))
      self.assertFalse(np.any(grad_ == 0))

  @test_util.run_deprecated_v1
  def test_float64_extreme_values_result_and_gradient_finite_and_nonzero(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): same as the float32 extreme-value test but with a 20 by 100
    # float64 grid from -200 to 700; asserts finite, non-zero values and gradients.
    with self.cached_session() as sess:
      # On the lower branch, log_cdf_laplace(x) = x, so we know this will be
      # fine, but test to -200 anyways.
      grid = _make_grid(
          np.float64, GridSpec(min=-200, max=700, shape=[20, 100]))
      grid = ops.convert_to_tensor(grid)

      actual = sm.log_cdf_laplace(grid)
      grad = gradients_impl.gradients(actual, grid)[0]

      actual_, grad_ = self.evaluate([actual, grad])

      # isfinite checks for NaN and Inf.
      self.assertAllTrue(np.isfinite(actual_))
      self.assertAllTrue(np.isfinite(grad_))
      self.assertFalse(np.any(actual_ == 0))
      self.assertFalse(np.any(grad_ == 0))


if __name__ == "__main__":
  test.main()
