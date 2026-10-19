# [MODULE DOC]
# KNOWN ISSUE: the module docstring says "Tests for initializers", but this file tests the Normal
# distribution (and its softplus-scale variant and KL divergence), not initializers.
# Unit tests for the Normal distribution in the distributions ops package. Covers parameter and
# sample shape helpers, dtype of method outputs, log density, density, CDF, log CDF, survival
# function, log survival function, entropy, mean, mode, quantile, variance, standard deviation,
# sampling statistics, reparameterized gradients, validation errors, shapes with placeholders,
# finite gradients at difficult points and the registered Normal-to-Normal KL divergence.
# Import-time side effect: optional scipy import (a warning is logged if it is missing).
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
"""Tests for initializers."""

import importlib
import math

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def try_import(name):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: optional-dependency loader used to import scipy for the comparison tests.
  # Takes a module name string, imports it dynamically and returns the module, or the null value
  # if the import fails (after logging a warning through the TensorFlow logger).
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module

stats = try_import("scipy.stats")


class NormalTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the Normal distribution family. Shared fixture: a seeded numpy random state
  # created in the set-up method, used to draw random locations and scales. Provides a custom
  # finiteness assertion and private helpers for parameter shapes and quantile gradients. Many
  # tests run in both graph and eager modes.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: overrides the base test set-up without calling the parent implementation, so
    # base-class per-test initialization is skipped.
    # Creates a numpy random state seeded with 123 and stores it on the instance. Takes no inputs
    # and returns nothing.
    self._rng = np.random.RandomState(123)

  def assertAllFinite(self, tensor):
    # [FUNCTION DOC]
    # HELPER: custom assertion used by the finite-gradient tests.
    # Evaluates the given tensor and asserts every element is finite (not infinite or
    # not-a-number). Fails the test otherwise; returns nothing.
    is_finite = np.isfinite(self.evaluate(tensor))
    all_true = np.ones_like(is_finite, dtype=np.bool_)
    self.assertAllEqual(all_true, is_finite)

  def _testParamShapes(self, sample_shape, expected):
    # [FUNCTION DOC]
    # HELPER: shared body for the parameter-shapes test.
    # Takes a sample shape (list or tensor) and the expected shape. Asks the Normal class for the
    # parameter shapes of the location and scale, asserts both equal the expected shape, builds a
    # Normal from zero and one tensors of those shapes and asserts a single sample has the expected
    # shape. Returns nothing.
    param_shapes = normal_lib.Normal.param_shapes(sample_shape)
    mu_shape, sigma_shape = param_shapes["loc"], param_shapes["scale"]
    self.assertAllEqual(expected, self.evaluate(mu_shape))
    self.assertAllEqual(expected, self.evaluate(sigma_shape))
    mu = array_ops.zeros(mu_shape)
    sigma = array_ops.ones(sigma_shape)
    self.assertAllEqual(
        expected,
        self.evaluate(array_ops.shape(normal_lib.Normal(mu, sigma).sample())))

  def _testParamStaticShapes(self, sample_shape, expected):
    # [FUNCTION DOC]
    # HELPER: shared body for the static parameter-shapes test.
    # Takes a sample shape (list or shape object) and the expected shape. Asks the Normal class for
    # the static parameter shapes of the location and scale and asserts both equal the expected
    # shape. Returns nothing.
    param_shapes = normal_lib.Normal.param_static_shapes(sample_shape)
    mu_shape, sigma_shape = param_shapes["loc"], param_shapes["scale"]
    self.assertEqual(expected, mu_shape)
    self.assertEqual(expected, sigma_shape)

  @test_util.run_in_graph_and_eager_modes
  def testSampleLikeArgsGetDistDType(self):
    # [FUNCTION DOC]
    # Builds a Normal from Python floats, asserts its dtype is 32-bit float, and asserts that each
    # of the density-related methods (log prob, prob, log CDF, CDF, log survival, survival,
    # quantile) called with a Python integer returns a 32-bit float result (the argument is cast to
    # the distribution dtype).
    dist = normal_lib.Normal(0., 1.)
    self.assertEqual(dtypes.float32, dist.dtype)
    for method in ("log_prob", "prob", "log_cdf", "cdf",
                   "log_survival_function", "survival_function", "quantile"):
      self.assertEqual(dtypes.float32, getattr(dist, method)(1).dtype)

  @test_util.run_in_graph_and_eager_modes
  def testParamShapes(self):
    # [FUNCTION DOC]
    # Runs the shared parameter-shapes check with a sample shape given as a list and as a constant
    # tensor, both expecting the same shape.
    sample_shape = [10, 3, 4]
    self._testParamShapes(sample_shape, sample_shape)
    self._testParamShapes(constant_op.constant(sample_shape), sample_shape)

  @test_util.run_in_graph_and_eager_modes
  def testParamStaticShapes(self):
    # [FUNCTION DOC]
    # Runs the shared static parameter-shapes check with a sample shape given as a list and as a
    # shape object, both expecting the same shape.
    sample_shape = [10, 3, 4]
    self._testParamStaticShapes(sample_shape, sample_shape)
    self._testParamStaticShapes(
        tensor_shape.TensorShape(sample_shape), sample_shape)

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testNormalWithSoftplusScale(self):
    # [FUNCTION DOC]
    # Builds the softplus-scale Normal from zero locations and a raw scale of minus two (ten by
    # three). Asserts the location is unchanged and the scale equals softplus of the raw scale.
    # Also checks that no eager garbage is left behind.
    mu = array_ops.zeros((10, 3))
    rho = array_ops.ones((10, 3)) * -2.
    normal = normal_lib.NormalWithSoftplusScale(loc=mu, scale=rho)
    self.assertAllEqual(self.evaluate(mu), self.evaluate(normal.loc))
    self.assertAllEqual(
        self.evaluate(nn_ops.softplus(rho)), self.evaluate(normal.scale))

  @test_util.run_in_graph_and_eager_modes
  def testNormalLogPDF(self):
    # [FUNCTION DOC]
    # Evaluates log density and density of a Normal (batch of six) at six points. Asserts the
    # output shapes (static and evaluated) equal the batch shape and, if scipy exists, that values
    # match scipy.
    batch_size = 6
    mu = constant_op.constant([3.0] * batch_size)
    sigma = constant_op.constant([math.sqrt(10.0)] * batch_size)
    x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
    normal = normal_lib.Normal(loc=mu, scale=sigma)

    log_pdf = normal.log_prob(x)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), log_pdf.get_shape())
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(log_pdf).shape)
    self.assertAllEqual(normal.batch_shape, log_pdf.get_shape())
    self.assertAllEqual(normal.batch_shape, self.evaluate(log_pdf).shape)

    pdf = normal.prob(x)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), pdf.get_shape())
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(pdf).shape)
    self.assertAllEqual(normal.batch_shape, pdf.get_shape())
    self.assertAllEqual(normal.batch_shape, self.evaluate(pdf).shape)

    if not stats:
      return
    expected_log_pdf = stats.norm(self.evaluate(mu),
                                  self.evaluate(sigma)).logpdf(x)
    self.assertAllClose(expected_log_pdf, self.evaluate(log_pdf))
    self.assertAllClose(np.exp(expected_log_pdf), self.evaluate(pdf))

  @test_util.run_in_graph_and_eager_modes
  def testNormalLogPDFMultidimensional(self):
    # [FUNCTION DOC]
    # Evaluates log density and density for a six-by-two batch at a column of six points. Asserts
    # shape six by two, equality with the batch shape (static and evaluated) and, if scipy exists,
    # agreement with scipy.
    batch_size = 6
    mu = constant_op.constant([[3.0, -3.0]] * batch_size)
    sigma = constant_op.constant(
        [[math.sqrt(10.0), math.sqrt(15.0)]] * batch_size)
    x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
    normal = normal_lib.Normal(loc=mu, scale=sigma)

    log_pdf = normal.log_prob(x)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.get_shape(), (6, 2))
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), log_pdf.get_shape())
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(log_pdf).shape)
    self.assertAllEqual(normal.batch_shape, log_pdf.get_shape())
    self.assertAllEqual(normal.batch_shape, self.evaluate(log_pdf).shape)

    pdf = normal.prob(x)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.get_shape(), (6, 2))
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), pdf.get_shape())
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), pdf_values.shape)
    self.assertAllEqual(normal.batch_shape, pdf.get_shape())
    self.assertAllEqual(normal.batch_shape, pdf_values.shape)

    if not stats:
      return
    expected_log_pdf = stats.norm(self.evaluate(mu),
                                  self.evaluate(sigma)).logpdf(x)
    self.assertAllClose(expected_log_pdf, log_pdf_values)
    self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  @test_util.run_in_graph_and_eager_modes
  def testNormalCDF(self):
    # [FUNCTION DOC]
    # Evaluates the CDF of a Normal with random location and scale (batch of fifty, seeded
    # fixture) at a grid from minus eight to eight. Asserts output shape equals batch shape and,
    # if scipy exists, exact agreement with scipy (zero absolute tolerance).
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)

    normal = normal_lib.Normal(loc=mu, scale=sigma)
    cdf = normal.cdf(x)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), cdf.get_shape())
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(cdf).shape)
    self.assertAllEqual(normal.batch_shape, cdf.get_shape())
    self.assertAllEqual(normal.batch_shape, self.evaluate(cdf).shape)
    if not stats:
      return
    expected_cdf = stats.norm(mu, sigma).cdf(x)
    self.assertAllClose(expected_cdf, self.evaluate(cdf), atol=0)

  @test_util.run_in_graph_and_eager_modes
  def testNormalSurvivalFunction(self):
    # [FUNCTION DOC]
    # Evaluates the survival function for a fifty-member random Normal on a grid from minus eight
    # to eight. Asserts output shape equals batch shape and, if scipy exists, agreement with scipy
    # (zero absolute tolerance).
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)

    normal = normal_lib.Normal(loc=mu, scale=sigma)

    sf = normal.survival_function(x)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), sf.get_shape())
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(sf).shape)
    self.assertAllEqual(normal.batch_shape, sf.get_shape())
    self.assertAllEqual(normal.batch_shape, self.evaluate(sf).shape)
    if not stats:
      return
    expected_sf = stats.norm(mu, sigma).sf(x)
    self.assertAllClose(expected_sf, self.evaluate(sf), atol=0)

  @test_util.run_in_graph_and_eager_modes
  def testNormalLogCDF(self):
    # [FUNCTION DOC]
    # Evaluates the log CDF for a fifty-member random Normal on a grid from minus one hundred to
    # ten (deep tail). Asserts output shape equals batch shape and, if scipy exists, agreement
    # with scipy at relative tolerance 0.001.
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-100.0, 10.0, batch_size).astype(np.float64)

    normal = normal_lib.Normal(loc=mu, scale=sigma)

    cdf = normal.log_cdf(x)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), cdf.get_shape())
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(cdf).shape)
    self.assertAllEqual(normal.batch_shape, cdf.get_shape())
    self.assertAllEqual(normal.batch_shape, self.evaluate(cdf).shape)

    if not stats:
      return
    expected_cdf = stats.norm(mu, sigma).logcdf(x)
    self.assertAllClose(expected_cdf, self.evaluate(cdf), atol=0, rtol=1e-3)

  def testFiniteGradientAtDifficultPoints(self):
    # [FUNCTION DOC]
    # For both 32-bit and 64-bit floats, builds a Normal from variables in a fresh graph and, for
    # CDF, log CDF, survival function, log survival function, log prob and prob evaluated at
    # extreme points (from minus one hundred to one hundred), asserts that the value and the
    # gradients with respect to location and scale are all finite after variable initialization.
    for dtype in [np.float32, np.float64]:
      g = ops.Graph()
      with g.as_default():
        mu = variables.Variable(dtype(0.0))
        sigma = variables.Variable(dtype(1.0))
        dist = normal_lib.Normal(loc=mu, scale=sigma)
        x = np.array([-100., -20., -5., 0., 5., 20., 100.]).astype(dtype)
        for func in [
            dist.cdf, dist.log_cdf, dist.survival_function,
            dist.log_survival_function, dist.log_prob, dist.prob
        ]:
          value = func(x)
          grads = gradients_impl.gradients(value, [mu, sigma])
          with self.session(graph=g):
            self.evaluate(variables.global_variables_initializer())
            self.assertAllFinite(value)
            self.assertAllFinite(grads[0])
            self.assertAllFinite(grads[1])

  @test_util.run_in_graph_and_eager_modes
  def testNormalLogSurvivalFunction(self):
    # [FUNCTION DOC]
    # Evaluates the log survival function for a fifty-member random Normal on a grid from minus ten
    # to one hundred. Asserts output shape equals batch shape and, if scipy exists, agreement with
    # scipy at relative tolerance 1e-5.
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-10.0, 100.0, batch_size).astype(np.float64)

    normal = normal_lib.Normal(loc=mu, scale=sigma)

    sf = normal.log_survival_function(x)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), sf.get_shape())
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(sf).shape)
    self.assertAllEqual(normal.batch_shape, sf.get_shape())
    self.assertAllEqual(normal.batch_shape, self.evaluate(sf).shape)

    if not stats:
      return
    expected_sf = stats.norm(mu, sigma).logsf(x)
    self.assertAllClose(expected_sf, self.evaluate(sf), atol=0, rtol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def testNormalEntropyWithScalarInputs(self):
    # [FUNCTION DOC]
    # Builds a Normal from scalar location and scale, asserts the entropy shape equals the batch
    # shape (static and evaluated) and, if scipy exists, that the entropy matches scipy.
    # Scipy.stats.norm cannot deal with the shapes in the other test.
    mu_v = 2.34
    sigma_v = 4.56
    normal = normal_lib.Normal(loc=mu_v, scale=sigma_v)

    entropy = normal.entropy()
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), entropy.get_shape())
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(entropy).shape)
    self.assertAllEqual(normal.batch_shape, entropy.get_shape())
    self.assertAllEqual(normal.batch_shape, self.evaluate(entropy).shape)
    # scipy.stats.norm cannot deal with these shapes.
    if not stats:
      return
    expected_entropy = stats.norm(mu_v, sigma_v).entropy()
    self.assertAllClose(expected_entropy, self.evaluate(entropy))

  @test_util.run_in_graph_and_eager_modes
  def testNormalEntropy(self):
    # [FUNCTION DOC]
    # Builds a Normal whose location (three) and scale (column of three) broadcast to a three by
    # three batch. Asserts the entropy equals the closed-form half log of two pi e variance and
    # that its shape equals the batch shape. Does not use scipy (it cannot handle these shapes).
    mu_v = np.array([1.0, 1.0, 1.0])
    sigma_v = np.array([[1.0, 2.0, 3.0]]).T
    normal = normal_lib.Normal(loc=mu_v, scale=sigma_v)

    # scipy.stats.norm cannot deal with these shapes.
    sigma_broadcast = mu_v * sigma_v
    expected_entropy = 0.5 * np.log(2 * np.pi * np.exp(1) * sigma_broadcast**2)
    entropy = normal.entropy()
    np.testing.assert_allclose(expected_entropy, self.evaluate(entropy))
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), entropy.get_shape())
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(entropy).shape)
    self.assertAllEqual(normal.batch_shape, entropy.get_shape())
    self.assertAllEqual(normal.batch_shape, self.evaluate(entropy).shape)

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testNormalMeanAndMode(self):
    # [FUNCTION DOC]
    # Builds a Normal with a one-element location broadcast against three scales. Asserts mean and
    # mode both have static shape three and evaluate to the location repeated. Also checks that no
    # eager garbage is left behind.
    # Mu will be broadcast to [7, 7, 7].
    mu = [7.]
    sigma = [11., 12., 13.]

    normal = normal_lib.Normal(loc=mu, scale=sigma)

    self.assertAllEqual((3,), normal.mean().get_shape())
    self.assertAllEqual([7., 7, 7], self.evaluate(normal.mean()))

    self.assertAllEqual((3,), normal.mode().get_shape())
    self.assertAllEqual([7., 7, 7], self.evaluate(normal.mode()))

  @test_util.run_in_graph_and_eager_modes
  def testNormalQuantile(self):
    # [FUNCTION DOC]
    # Evaluates the quantile function of a fifty-two-member random Normal at probabilities from
    # zero to one plus two extreme tail values (to hit all pieces of the rational approximation).
    # Asserts output shape equals batch shape and, if scipy exists, agreement with the scipy
    # percent point function.
    batch_size = 52
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    p = np.linspace(0., 1.0, batch_size - 2).astype(np.float64)
    # Quantile performs piecewise rational approximation so adding some
    # special input values to make sure we hit all the pieces.
    p = np.hstack((p, np.exp(-33), 1. - np.exp(-33)))

    normal = normal_lib.Normal(loc=mu, scale=sigma)
    x = normal.quantile(p)

    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), x.get_shape())
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(x).shape)
    self.assertAllEqual(normal.batch_shape, x.get_shape())
    self.assertAllEqual(normal.batch_shape, self.evaluate(x).shape)

    if not stats:
      return
    expected_x = stats.norm(mu, sigma).ppf(p)
    self.assertAllClose(expected_x, self.evaluate(x), atol=0.)

  def _baseQuantileFiniteGradientAtDifficultPoints(self, dtype):
    # [FUNCTION DOC]
    # HELPER: shared body for the quantile finite-gradient tests.
    # Takes a numpy float type. In a fresh graph builds a Normal from variables and a variable
    # holding extreme probabilities (zero, tiny, moderate, near one, and one), computes the
    # quantile and, after variable initialization, asserts gradients with respect to location and
    # to the probabilities are finite. Returns nothing.
    g = ops.Graph()
    with g.as_default():
      mu = variables.Variable(dtype(0.0))
      sigma = variables.Variable(dtype(1.0))
      dist = normal_lib.Normal(loc=mu, scale=sigma)
      p = variables.Variable(
          np.array([0.,
                    np.exp(-32.), np.exp(-2.),
                    1. - np.exp(-2.), 1. - np.exp(-32.),
                    1.]).astype(dtype))

      value = dist.quantile(p)
      grads = gradients_impl.gradients(value, [mu, p])
      with self.cached_session(graph=g):
        self.evaluate(variables.global_variables_initializer())
        self.assertAllFinite(grads[0])
        self.assertAllFinite(grads[1])

  def testQuantileFiniteGradientAtDifficultPointsFloat32(self):
    # [FUNCTION DOC]
    # Runs the shared quantile finite-gradient check with 32-bit floats.
    self._baseQuantileFiniteGradientAtDifficultPoints(np.float32)

  def testQuantileFiniteGradientAtDifficultPointsFloat64(self):
    # [FUNCTION DOC]
    # Runs the shared quantile finite-gradient check with 64-bit floats.
    self._baseQuantileFiniteGradientAtDifficultPoints(np.float64)

  @test_util.run_in_graph_and_eager_modes
  def testNormalVariance(self):
    # [FUNCTION DOC]
    # Builds a Normal with three locations and one scale of seven (broadcast). Asserts variance has
    # static shape three and equals forty-nine for each member.
    # sigma will be broadcast to [7, 7, 7]
    mu = [1., 2., 3.]
    sigma = [7.]

    normal = normal_lib.Normal(loc=mu, scale=sigma)

    self.assertAllEqual((3,), normal.variance().get_shape())
    self.assertAllEqual([49., 49, 49], self.evaluate(normal.variance()))

  @test_util.run_in_graph_and_eager_modes
  def testNormalStandardDeviation(self):
    # [FUNCTION DOC]
    # Builds a Normal with three locations and one scale of seven (broadcast). Asserts standard
    # deviation has static shape three and equals seven for each member.
    # sigma will be broadcast to [7, 7, 7]
    mu = [1., 2., 3.]
    sigma = [7.]

    normal = normal_lib.Normal(loc=mu, scale=sigma)

    self.assertAllEqual((3,), normal.stddev().get_shape())
    self.assertAllEqual([7., 7, 7], self.evaluate(normal.stddev()))

  @test_util.run_in_graph_and_eager_modes
  def testNormalSample(self):
    # [FUNCTION DOC]
    # Draws 100000 samples (count given as a tensor, no seed) from a scalar Normal (mean 3,
    # standard deviation root three). Asserts sample shape, sample mean and standard deviation
    # within 0.1 of the true values, and that the sample shape equals the count followed by the
    # batch shape (computed from both the batch shape tensor and static batch shape).
    mu = constant_op.constant(3.0)
    sigma = constant_op.constant(math.sqrt(3.0))
    mu_v = 3.0
    sigma_v = np.sqrt(3.0)
    n = constant_op.constant(100000)
    normal = normal_lib.Normal(loc=mu, scale=sigma)
    samples = normal.sample(n)
    sample_values = self.evaluate(samples)
    # Note that the standard error for the sample mean is ~ sigma / sqrt(n).
    # The sample variance similarly is dependent on sigma and n.
    # Thus, the tolerances below are very sensitive to number of samples
    # as well as the variances chosen.
    self.assertEqual(sample_values.shape, (100000,))
    self.assertAllClose(sample_values.mean(), mu_v, atol=1e-1)
    self.assertAllClose(sample_values.std(), sigma_v, atol=1e-1)

    expected_samples_shape = tensor_shape.TensorShape(
        [self.evaluate(n)]).concatenate(
            tensor_shape.TensorShape(
                self.evaluate(normal.batch_shape_tensor())))

    self.assertAllEqual(expected_samples_shape, samples.get_shape())
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

    expected_samples_shape = (
        tensor_shape.TensorShape([self.evaluate(n)]).concatenate(
            normal.batch_shape))

    self.assertAllEqual(expected_samples_shape, samples.get_shape())
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

  def testNormalFullyReparameterized(self):
    # [FUNCTION DOC]
    # Samples a Normal inside a gradient tape watching location and scale and asserts both
    # gradients of the samples exist (are not null).
    mu = constant_op.constant(4.0)
    sigma = constant_op.constant(3.0)
    with backprop.GradientTape() as tape:
      tape.watch(mu)
      tape.watch(sigma)
      normal = normal_lib.Normal(loc=mu, scale=sigma)
      samples = normal.sample(100)
    grad_mu, grad_sigma = tape.gradient(samples, [mu, sigma])
    self.assertIsNotNone(grad_mu)
    self.assertIsNotNone(grad_sigma)

  @test_util.run_in_graph_and_eager_modes
  def testNormalSampleMultiDimensional(self):
    # [FUNCTION DOC]
    # Draws 100000 unseeded samples from a Normal with a two-by-two batch of locations and scales.
    # Asserts static sample shape and that the mean and standard deviation of the first batch row
    # are within 0.1 of the true values for both columns. Also asserts the sample shape equals the
    # count followed by the batch shape (from tensor and static forms).
    batch_size = 2
    mu = constant_op.constant([[3.0, -3.0]] * batch_size)
    sigma = constant_op.constant(
        [[math.sqrt(2.0), math.sqrt(3.0)]] * batch_size)
    mu_v = [3.0, -3.0]
    sigma_v = [np.sqrt(2.0), np.sqrt(3.0)]
    n = constant_op.constant(100000)
    normal = normal_lib.Normal(loc=mu, scale=sigma)
    samples = normal.sample(n)
    sample_values = self.evaluate(samples)
    # Note that the standard error for the sample mean is ~ sigma / sqrt(n).
    # The sample variance similarly is dependent on sigma and n.
    # Thus, the tolerances below are very sensitive to number of samples
    # as well as the variances chosen.
    self.assertEqual(samples.get_shape(), (100000, batch_size, 2))
    self.assertAllClose(sample_values[:, 0, 0].mean(), mu_v[0], atol=1e-1)
    self.assertAllClose(sample_values[:, 0, 0].std(), sigma_v[0], atol=1e-1)
    self.assertAllClose(sample_values[:, 0, 1].mean(), mu_v[1], atol=1e-1)
    self.assertAllClose(sample_values[:, 0, 1].std(), sigma_v[1], atol=1e-1)

    expected_samples_shape = tensor_shape.TensorShape(
        [self.evaluate(n)]).concatenate(
            tensor_shape.TensorShape(
                self.evaluate(normal.batch_shape_tensor())))
    self.assertAllEqual(expected_samples_shape, samples.get_shape())
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

    expected_samples_shape = (
        tensor_shape.TensorShape([self.evaluate(n)]).concatenate(
            normal.batch_shape))
    self.assertAllEqual(expected_samples_shape, samples.get_shape())
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

  @test_util.run_in_graph_and_eager_modes
  def testNegativeSigmaFails(self):
    # [FUNCTION DOC]
    # With argument validation on, builds a Normal with a negative scale and asserts that
    # evaluating its mean raises an op error saying the positivity condition did not hold.
    with self.assertRaisesOpError("Condition x > 0 did not hold"):
      normal = normal_lib.Normal(
          loc=[1.], scale=[-5.], validate_args=True, name="G")
      self.evaluate(normal.mean())

  @test_util.run_in_graph_and_eager_modes
  def testNormalShape(self):
    # [FUNCTION DOC]
    # Builds a Normal with a batch of five locations and a scalar scale. Asserts batch shape
    # (tensor and static) is five and event shape (tensor and static) is empty.
    mu = constant_op.constant([-3.0] * 5)
    sigma = constant_op.constant(11.0)
    normal = normal_lib.Normal(loc=mu, scale=sigma)

    self.assertEqual(self.evaluate(normal.batch_shape_tensor()), [5])
    self.assertEqual(normal.batch_shape, tensor_shape.TensorShape([5]))
    self.assertAllEqual(self.evaluate(normal.event_shape_tensor()), [])
    self.assertEqual(normal.event_shape, tensor_shape.TensorShape([]))

  @test_util.run_deprecated_v1
  def testNormalShapeWithPlaceholders(self):
    # [FUNCTION DOC]
    # Builds a Normal from two unshaped float placeholders. Asserts the static batch shape is
    # fully unknown, the static event shape is empty, the event shape tensor is empty, and that
    # feeding a scalar location and a two-element scale gives a batch shape tensor of two. Graph
    # mode only (deprecated-v1 decorator).
    mu = array_ops.placeholder(dtype=dtypes.float32)
    sigma = array_ops.placeholder(dtype=dtypes.float32)
    normal = normal_lib.Normal(loc=mu, scale=sigma)

    with self.cached_session() as sess:
      # get_batch_shape should return an "<unknown>" tensor.
      self.assertEqual(normal.batch_shape, tensor_shape.TensorShape(None))
      self.assertEqual(normal.event_shape, ())
      self.assertAllEqual(self.evaluate(normal.event_shape_tensor()), [])
      self.assertAllEqual(
          sess.run(normal.batch_shape_tensor(),
                   feed_dict={mu: 5.0,
                              sigma: [1.0, 2.0]}), [2])

  @test_util.run_in_graph_and_eager_modes
  def testNormalNormalKL(self):
    # [FUNCTION DOC]
    # Computes the registered KL divergence between two six-member Normals and asserts the result
    # has shape six and matches the closed-form Gaussian KL formula.
    batch_size = 6
    mu_a = np.array([3.0] * batch_size)
    sigma_a = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    mu_b = np.array([-3.0] * batch_size)
    sigma_b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    n_a = normal_lib.Normal(loc=mu_a, scale=sigma_a)
    n_b = normal_lib.Normal(loc=mu_b, scale=sigma_b)

    kl = kullback_leibler.kl_divergence(n_a, n_b)
    kl_val = self.evaluate(kl)

    kl_expected = ((mu_a - mu_b)**2 / (2 * sigma_b**2) + 0.5 * (
        (sigma_a**2 / sigma_b**2) - 1 - 2 * np.log(sigma_a / sigma_b)))

    self.assertEqual(kl.get_shape(), (batch_size,))
    self.assertAllClose(kl_val, kl_expected)


if __name__ == "__main__":
  test.main()
