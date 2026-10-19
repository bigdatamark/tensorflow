# [MODULE DOC]
# Unit tests for the Student t distribution class in the distributions ops package, including
# its subclass that applies an absolute-value/floor transform to the degrees of freedom and a
# softplus to the scale. Covers density and cumulative functions, entropy, sampling (statistics,
# seeding, small degrees of freedom, reparameterization), shape broadcasting, mean/variance/
# standard deviation/mode behavior including undefined values and the allow-nan-stats switch, and
# argument validation. Reference values come from scipy.stats when it can be imported; otherwise
# those comparisons are skipped. Import-time side effect: attempts the optional scipy import and
# logs a warning on failure. Runs as a test binary through the TensorFlow test main.
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
"""Tests for Student t distribution."""

import importlib
import math

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import student_t
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def try_import(name):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: optional-dependency loader used to import scipy for reference values.
  # Imports the module with the given dotted name and returns it. On an import error it logs a
  # warning (side effect) naming the module and the error and returns the null value instead of
  # raising. Other exceptions propagate.
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


stats = try_import("scipy.stats")


@test_util.run_all_in_graph_and_eager_modes
class StudentTTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the Student t distribution (and its abs-df/softplus-scale variant). The
  # class decorator runs every test in both graph and eager modes. No shared fixtures: each test
  # builds its own distribution from constants. Uses private helpers for a KL-divergence
  # histogram comparison with scipy samples and for a numerical integral of a density.

  def testStudentPDFAndLogPDF(self):
    # [FUNCTION DOC]
    # Test: builds a batch of six Student t distributions with degrees of freedom 3, location 7
    # and a negative scale (the magnitude 8 is what scipy is given, so the sign is expected to be
    # irrelevant), evaluates log density and density at six points, asserts both have static
    # shape of six, and (when scipy is present) that they match scipy's values and are mutually
    # consistent under log/exp.
    batch_size = 6
    df = constant_op.constant([3.] * batch_size)
    mu = constant_op.constant([7.] * batch_size)
    sigma = constant_op.constant([8.] * batch_size)
    df_v = 3.
    mu_v = 7.
    sigma_v = 8.
    t = np.array([-2.5, 2.5, 8., 0., -1., 2.], dtype=np.float32)
    student = student_t.StudentT(df, loc=mu, scale=-sigma)  # pylint: disable=invalid-unary-operand-type

    log_pdf = student.log_prob(t)
    self.assertEqual(log_pdf.get_shape(), (6,))
    log_pdf_values = self.evaluate(log_pdf)
    pdf = student.prob(t)
    self.assertEqual(pdf.get_shape(), (6,))
    pdf_values = self.evaluate(pdf)

    if not stats:
      return

    expected_log_pdf = stats.t.logpdf(t, df_v, loc=mu_v, scale=sigma_v)
    expected_pdf = stats.t.pdf(t, df_v, loc=mu_v, scale=sigma_v)
    self.assertAllClose(expected_log_pdf, log_pdf_values)
    self.assertAllClose(np.log(expected_pdf), log_pdf_values)
    self.assertAllClose(expected_pdf, pdf_values)
    self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  def testStudentLogPDFMultidimensional(self):
    # [FUNCTION DOC]
    # Test: two-column batch (degrees of freedom 1.5 and 7.2, locations 3 and -3, scales with one
    # negative entry) evaluated at a column of six points; asserts the log density and density
    # have static shape six by two and match scipy (log/exp consistent) when scipy is available.
    batch_size = 6
    df = constant_op.constant([[1.5, 7.2]] * batch_size)
    mu = constant_op.constant([[3., -3.]] * batch_size)
    sigma = constant_op.constant(
        [[-math.sqrt(10.), math.sqrt(15.)]] * batch_size)
    df_v = np.array([1.5, 7.2])
    mu_v = np.array([3., -3.])
    sigma_v = np.array([np.sqrt(10.), np.sqrt(15.)])
    t = np.array([[-2.5, 2.5, 4., 0., -1., 2.]], dtype=np.float32).T
    student = student_t.StudentT(df, loc=mu, scale=sigma)
    log_pdf = student.log_prob(t)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.get_shape(), (6, 2))
    pdf = student.prob(t)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.get_shape(), (6, 2))

    if not stats:
      return
    expected_log_pdf = stats.t.logpdf(t, df_v, loc=mu_v, scale=sigma_v)
    expected_pdf = stats.t.pdf(t, df_v, loc=mu_v, scale=sigma_v)
    self.assertAllClose(expected_log_pdf, log_pdf_values)
    self.assertAllClose(np.log(expected_pdf), log_pdf_values)
    self.assertAllClose(expected_pdf, pdf_values)
    self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  def testStudentCDFAndLogCDF(self):
    # [FUNCTION DOC]
    # Test: batch of six distributions (degrees of freedom 3, location 7, negative scale of
    # magnitude 8) evaluated at six points; asserts log CDF and CDF have static shape six and,
    # with scipy available, match scipy's values with relative tolerance 1e-5 and are consistent
    # under log/exp.
    batch_size = 6
    df = constant_op.constant([3.] * batch_size)
    mu = constant_op.constant([7.] * batch_size)
    sigma = constant_op.constant([-8.] * batch_size)
    df_v = 3.
    mu_v = 7.
    sigma_v = 8.
    t = np.array([-2.5, 2.5, 8., 0., -1., 2.], dtype=np.float32)
    student = student_t.StudentT(df, loc=mu, scale=sigma)

    log_cdf = student.log_cdf(t)
    self.assertEqual(log_cdf.get_shape(), (6,))
    log_cdf_values = self.evaluate(log_cdf)
    cdf = student.cdf(t)
    self.assertEqual(cdf.get_shape(), (6,))
    cdf_values = self.evaluate(cdf)

    if not stats:
      return
    expected_log_cdf = stats.t.logcdf(t, df_v, loc=mu_v, scale=sigma_v)
    expected_cdf = stats.t.cdf(t, df_v, loc=mu_v, scale=sigma_v)
    self.assertAllClose(expected_log_cdf, log_cdf_values, atol=0., rtol=1e-5)
    self.assertAllClose(
        np.log(expected_cdf), log_cdf_values, atol=0., rtol=1e-5)
    self.assertAllClose(expected_cdf, cdf_values, atol=0., rtol=1e-5)
    self.assertAllClose(
        np.exp(expected_log_cdf), cdf_values, atol=0., rtol=1e-5)

  def testStudentEntropy(self):
    # [FUNCTION DOC]
    # Test: parameters are a one-by-three row for degrees of freedom and location and a
    # three-by-one column for scale (one negative), so they broadcast to a three-by-three batch.
    # Asserts entropy matches scipy's entropy computed on explicitly broadcast flattened inputs
    # (scale by magnitude), reshaped back. Skipped after evaluation when scipy is unavailable.
    df_v = np.array([[2., 3., 7.]])  # 1x3
    mu_v = np.array([[1., -1, 0]])  # 1x3
    sigma_v = np.array([[1., -2., 3.]]).T  # transposed => 3x1
    student = student_t.StudentT(df=df_v, loc=mu_v, scale=sigma_v)
    ent = student.entropy()
    ent_values = self.evaluate(ent)

    # Help scipy broadcast to 3x3
    ones = np.array([[1, 1, 1]])
    sigma_bc = np.abs(sigma_v) * ones
    mu_bc = ones.T * mu_v
    df_bc = ones.T * df_v
    if not stats:
      return
    expected_entropy = stats.t.entropy(
        np.reshape(df_bc, [-1]),
        loc=np.reshape(mu_bc, [-1]),
        scale=np.reshape(sigma_bc, [-1]))
    expected_entropy = np.reshape(expected_entropy, df_bc.shape)
    self.assertAllClose(expected_entropy, ent_values)

  def testStudentSample(self):
    # [FUNCTION DOC]
    # Test: draws 200000 seeded samples from a scalar distribution (degrees of freedom 4,
    # location 3, negative scale of magnitude sqrt(10)); asserts the sample shape, that the
    # sample mean and variance are within 10 percent of the theoretical values, and that a
    # histogram-based KL approximation against scipy samples is small (via the private helper).
    df = constant_op.constant(4.)
    mu = constant_op.constant(3.)
    sigma = constant_op.constant(-math.sqrt(10.))
    df_v = 4.
    mu_v = 3.
    sigma_v = np.sqrt(10.)
    n = constant_op.constant(200000)
    student = student_t.StudentT(df=df, loc=mu, scale=sigma)
    samples = student.sample(n, seed=123456)
    sample_values = self.evaluate(samples)
    n_val = 200000
    self.assertEqual(sample_values.shape, (n_val,))
    self.assertAllClose(sample_values.mean(), mu_v, rtol=0.1, atol=0)
    self.assertAllClose(
        sample_values.var(), sigma_v**2 * df_v / (df_v - 2), rtol=0.1, atol=0)
    self._checkKLApprox(df_v, mu_v, sigma_v, sample_values)

  # Test that sampling with the same seed twice gives the same results.
  def testStudentSampleMultipleTimes(self):
    # [FUNCTION DOC]
    # Test: sets the global random seed, creates a distribution and draws 100 samples with a
    # fixed op seed; resets the global seed, creates a second identically parameterized
    # distribution under a different name and draws again with the same op seed. Asserts the
    # two sample sets are equal. Side effect: modifies the global random seed.
    df = constant_op.constant(4.)
    mu = constant_op.constant(3.)
    sigma = constant_op.constant(math.sqrt(10.))
    n = constant_op.constant(100)

    random_seed.set_random_seed(654321)
    student = student_t.StudentT(df=df, loc=mu, scale=sigma, name="student_t1")
    samples1 = self.evaluate(student.sample(n, seed=123456))

    random_seed.set_random_seed(654321)
    student2 = student_t.StudentT(df=df, loc=mu, scale=sigma, name="student_t2")
    samples2 = self.evaluate(student2.sample(n, seed=123456))

    self.assertAllClose(samples1, samples2)

  def testStudentSampleSmallDfNoNan(self):
    # [FUNCTION DOC]
    # Test: draws 200000 seeded samples from distributions with extremely small degrees of
    # freedom (0.1 down to 1e-20, location 1, scale 1) and asserts the result has shape
    # 200000 by 4 and contains no NaN values.
    df_v = [1e-1, 1e-5, 1e-10, 1e-20]
    df = constant_op.constant(df_v)
    n = constant_op.constant(200000)
    student = student_t.StudentT(df=df, loc=1., scale=1.)
    samples = student.sample(n, seed=123456)
    sample_values = self.evaluate(samples)
    n_val = 200000
    self.assertEqual(sample_values.shape, (n_val, 4))
    self.assertTrue(np.all(np.logical_not(np.isnan(sample_values))))

  def testStudentSampleMultiDimensional(self):
    # [FUNCTION DOC]
    # Test: batch of seven by two distributions (degrees of freedom 5 and 7, locations 3 and -3,
    # scales sqrt(10) and sqrt(15)); draws 200000 samples and asserts the static sample shape,
    # that per-column sample mean (10 percent) and variance (20 percent) match theory for the
    # first batch row, and that the KL approximation check passes for each column.
    batch_size = 7
    df = constant_op.constant([[5., 7.]] * batch_size)
    mu = constant_op.constant([[3., -3.]] * batch_size)
    sigma = constant_op.constant(
        [[math.sqrt(10.), math.sqrt(15.)]] * batch_size)
    df_v = [5., 7.]
    mu_v = [3., -3.]
    sigma_v = [np.sqrt(10.), np.sqrt(15.)]
    n = constant_op.constant(200000)
    student = student_t.StudentT(df=df, loc=mu, scale=sigma)
    samples = student.sample(n, seed=123456)
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.get_shape(), (200000, batch_size, 2))
    self.assertAllClose(
        sample_values[:, 0, 0].mean(), mu_v[0], rtol=0.1, atol=0)
    self.assertAllClose(
        sample_values[:, 0, 0].var(),
        sigma_v[0]**2 * df_v[0] / (df_v[0] - 2),
        rtol=0.2,
        atol=0)
    self._checkKLApprox(df_v[0], mu_v[0], sigma_v[0], sample_values[:, 0, 0])
    self.assertAllClose(
        sample_values[:, 0, 1].mean(), mu_v[1], rtol=0.1, atol=0)
    self.assertAllClose(
        sample_values[:, 0, 1].var(),
        sigma_v[1]**2 * df_v[1] / (df_v[1] - 2),
        rtol=0.2,
        atol=0)
    self._checkKLApprox(df_v[1], mu_v[1], sigma_v[1], sample_values[:, 0, 1])

  def _checkKLApprox(self, df, mu, sigma, samples):
    # [FUNCTION DOC]
    # HELPER: statistical assertion shared by the sampling tests.
    # Inputs: scalar degrees of freedom, location and scale, and an array of samples from the
    # distribution under test. Seeds numpy's global generator (side effect), returns early if
    # scipy is unavailable, otherwise draws the same number of scipy samples, histograms both
    # over the central 99 percent interval with 100 bins, asserts both histograms cover at least
    # 98 percent of the samples, and asserts the approximate KL divergence between the
    # (add-one smoothed) normalized histograms is below 1.
    n = samples.size
    np.random.seed(137)
    if not stats:
      return
    sample_scipy = stats.t.rvs(df, loc=mu, scale=sigma, size=n)
    covg = 0.99
    r = stats.t.interval(covg, df, loc=mu, scale=sigma)
    bins = 100
    hist, _ = np.histogram(samples, bins=bins, range=r)
    hist_scipy, _ = np.histogram(sample_scipy, bins=bins, range=r)
    self.assertGreater(hist.sum(), n * (covg - .01))
    self.assertGreater(hist_scipy.sum(), n * (covg - .01))
    hist_min1 = hist + 1.  # put at least one item in each bucket
    hist_norm = hist_min1 / hist_min1.sum()
    hist_scipy_min1 = hist_scipy + 1.  # put at least one item in each bucket
    hist_scipy_norm = hist_scipy_min1 / hist_scipy_min1.sum()
    kl_appx = np.sum(np.log(hist_scipy_norm / hist_norm) * hist_scipy_norm)
    self.assertLess(kl_appx, 1)

  def testBroadcastingParams(self):
    # [FUNCTION DOC]
    # Test: builds distributions where exactly one of degrees of freedom, location or scale is a
    # length-three list and the others are scalars, and asserts (through a nested checker) that
    # mean, variance, entropy, log density and density have static shape three and that 37
    # samples have shape 37 by 3.

    def _check(student):
      # [FUNCTION DOC]
      # HELPER: nested assertion helper. Takes a Student t distribution and asserts the static
      # shapes of its mean, variance, entropy, log density and density at a scalar point (all
      # three) and of 37 samples (37 by 3). Returns nothing.
      self.assertEqual(student.mean().get_shape(), (3,))
      self.assertEqual(student.variance().get_shape(), (3,))
      self.assertEqual(student.entropy().get_shape(), (3,))
      self.assertEqual(student.log_prob(2.).get_shape(), (3,))
      self.assertEqual(student.prob(2.).get_shape(), (3,))
      self.assertEqual(student.sample(37).get_shape(), (37, 3,))

    _check(student_t.StudentT(df=[2., 3., 4.,], loc=2., scale=1.))
    _check(student_t.StudentT(df=7., loc=[2., 3., 4.,], scale=1.))
    _check(student_t.StudentT(df=7., loc=3., scale=[2., 3., 4.,]))

  def testBroadcastingPdfArgs(self):
    # [FUNCTION DOC]
    # Test: checks that the static shape of log density and density broadcasts the evaluation
    # point against the batch shape. Three batch layouts are used (flat of three, one-by-three
    # row, three-by-one column), each built three ways by varying which parameter is the list,
    # and each queried with a scalar, a vector, a row and a column of points via nested helpers.

    def _assert_shape(student, arg, shape):
      # [FUNCTION DOC]
      # HELPER: nested assertion helper. Asserts that both the log density and the density of
      # the given distribution at the given point(s) have the expected static shape.
      self.assertEqual(student.log_prob(arg).get_shape(), shape)
      self.assertEqual(student.prob(arg).get_shape(), shape)

    def _check(student):
      # [FUNCTION DOC]
      # HELPER: nested assertion helper for a batch of shape three. Queries a scalar (expects
      # shape three), a vector of three (three), a one-by-three row (one by three) and a
      # three-by-one column (three by three).
      _assert_shape(student, 2., (3,))
      xs = np.array([2., 3., 4.], dtype=np.float32)
      _assert_shape(student, xs, (3,))
      xs = np.array([xs])
      _assert_shape(student, xs, (1, 3))
      xs = xs.T
      _assert_shape(student, xs, (3, 3))

    _check(student_t.StudentT(df=[2., 3., 4.,], loc=2., scale=1.))
    _check(student_t.StudentT(df=7., loc=[2., 3., 4.,], scale=1.))
    _check(student_t.StudentT(df=7., loc=3., scale=[2., 3., 4.,]))

    def _check2d(student):
      # [FUNCTION DOC]
      # HELPER: nested assertion helper for a one-by-three batch. Expected shapes: scalar,
      # vector and row queries give one by three; a column query gives three by three.
      _assert_shape(student, 2., (1, 3))
      xs = np.array([2., 3., 4.], dtype=np.float32)
      _assert_shape(student, xs, (1, 3))
      xs = np.array([xs])
      _assert_shape(student, xs, (1, 3))
      xs = xs.T
      _assert_shape(student, xs, (3, 3))

    _check2d(student_t.StudentT(df=[[2., 3., 4.,]], loc=2., scale=1.))
    _check2d(student_t.StudentT(df=7., loc=[[2., 3., 4.,]], scale=1.))
    _check2d(student_t.StudentT(df=7., loc=3., scale=[[2., 3., 4.,]]))

    def _check2d_rows(student):
      # [FUNCTION DOC]
      # HELPER: nested assertion helper for a three-by-one batch. Expected shapes: scalar gives
      # three by one, vector and row queries give three by three, column query gives three by
      # one.
      _assert_shape(student, 2., (3, 1))
      xs = np.array([2., 3., 4.], dtype=np.float32)  # (3,)
      _assert_shape(student, xs, (3, 3))
      xs = np.array([xs])  # (1,3)
      _assert_shape(student, xs, (3, 3))
      xs = xs.T  # (3,1)
      _assert_shape(student, xs, (3, 1))

    _check2d_rows(student_t.StudentT(df=[[2.], [3.], [4.]], loc=2., scale=1.))
    _check2d_rows(student_t.StudentT(df=7., loc=[[2.], [3.], [4.]], scale=1.))
    _check2d_rows(student_t.StudentT(df=7., loc=3., scale=[[2.], [3.], [4.]]))

  def testMeanAllowNanStatsIsFalseWorksWhenAllBatchMembersAreDefined(self):
    # [FUNCTION DOC]
    # Test: with all degrees of freedom above one (3, 5, 7) the mean is defined, and the test
    # asserts that the mean equals the location values. Uses the default nan-stats setting
    # (disallowed) as the name suggests.
    mu = [1., 3.3, 4.4]
    student = student_t.StudentT(df=[3., 5., 7.], loc=mu, scale=[3., 2., 1.])
    mean = self.evaluate(student.mean())
    self.assertAllClose([1., 3.3, 4.4], mean)

  def testMeanAllowNanStatsIsFalseRaisesWhenBatchMemberIsUndefined(self):
    # [FUNCTION DOC]
    # Test: with nan statistics disallowed and a degrees-of-freedom entry of 0.5 (mean
    # undefined), evaluating the mean must raise an op error whose message mentions the failed
    # "x < y" assertion.
    mu = [1., 3.3, 4.4]
    student = student_t.StudentT(
        df=[0.5, 5., 7.], loc=mu, scale=[3., 2., 1.], allow_nan_stats=False)
    with self.assertRaisesOpError("x < y"):
      self.evaluate(student.mean())

  def testMeanAllowNanStatsIsTrueReturnsNaNForUndefinedBatchMembers(self):
    # [FUNCTION DOC]
    # Test: with nan statistics allowed and degrees of freedom 0.5, 1, 3, 5, 7, asserts the mean
    # is NaN for the first two (degrees of freedom at most one) and equals the location for the
    # rest.
    mu = [-2, 0., 1., 3.3, 4.4]
    sigma = [5., 4., 3., 2., 1.]
    student = student_t.StudentT(
        df=[0.5, 1., 3., 5., 7.], loc=mu, scale=sigma, allow_nan_stats=True)
    mean = self.evaluate(student.mean())
    self.assertAllClose([np.nan, np.nan, 1., 3.3, 4.4], mean)

  def testVarianceAllowNanStatsTrueReturnsNaNforUndefinedBatchMembers(self):
    # [FUNCTION DOC]
    # Test: with nan statistics allowed and degrees of freedom 0.5, 1.5, 3, 5, 7, evaluates the
    # variance and (when scipy is available) compares to scipy's variance for all entries except
    # the first, which is excluded because NaN versus infinity differs between scipy versions.
    # The comparison therefore does not itself assert NaN for the undefined entry despite the
    # name.
    # df = 0.5 ==> undefined mean ==> undefined variance.
    # df = 1.5 ==> infinite variance.
    df = [0.5, 1.5, 3., 5., 7.]
    mu = [-2, 0., 1., 3.3, 4.4]
    sigma = [5., 4., 3., 2., 1.]
    student = student_t.StudentT(
        df=df, loc=mu, scale=sigma, allow_nan_stats=True)
    var = self.evaluate(student.variance())

    if not stats:
      return
    expected_var = [
        stats.t.var(d, loc=m, scale=s) for (d, m, s) in zip(df, mu, sigma)
    ]
    # Slicing off first element due to nan/inf mismatch in different SciPy
    # versions.
    self.assertAllClose(expected_var[1:], var[1:])

  def testVarianceAllowNanStatsFalseGivesCorrectValueForDefinedBatchMembers(
      self):
    # [FUNCTION DOC]
    # Test: with degrees of freedom 1.5 (infinite variance), 3, 5, 7 and the default nan-stats
    # setting, evaluates the variance and, with scipy available, asserts equality with scipy's
    # variance including the infinite entry.
    # df = 1.5 ==> infinite variance.
    df = [1.5, 3., 5., 7.]
    mu = [0., 1., 3.3, 4.4]
    sigma = [4., 3., 2., 1.]
    student = student_t.StudentT(df=df, loc=mu, scale=sigma)
    var = self.evaluate(student.variance())

    if not stats:
      return
    expected_var = [
        stats.t.var(d, loc=m, scale=s) for (d, m, s) in zip(df, mu, sigma)
    ]
    self.assertAllClose(expected_var, var)

  def testVarianceAllowNanStatsFalseRaisesForUndefinedBatchMembers(self):
    # [FUNCTION DOC]
    # Test: with nan statistics disallowed, degrees of freedom 1 and then 0.5 each make the
    # variance undefined; evaluating the variance must raise an op error mentioning the failed
    # "x < y" assertion in both cases.
    # df <= 1 ==> variance not defined
    student = student_t.StudentT(df=1., loc=0., scale=1., allow_nan_stats=False)
    with self.assertRaisesOpError("x < y"):
      self.evaluate(student.variance())

    # df <= 1 ==> variance not defined
    student = student_t.StudentT(
        df=0.5, loc=0., scale=1., allow_nan_stats=False)
    with self.assertRaisesOpError("x < y"):
      self.evaluate(student.variance())

  def testStd(self):
    # [FUNCTION DOC]
    # Test: five distributions with degrees of freedom above two, a single-element location list
    # broadcast across them and varying scales. Asserts the standard deviation matches scipy's
    # (when available); the location list is replicated only after evaluation, to feed scipy.
    # Defined for all batch members.
    df = [3.5, 5., 3., 5., 7.]
    mu = [-2.2]
    sigma = [5., 4., 3., 2., 1.]
    student = student_t.StudentT(df=df, loc=mu, scale=sigma)
    # Test broadcast of mu across shape of df/sigma
    stddev = self.evaluate(student.stddev())
    mu *= len(df)

    if not stats:
      return
    expected_stddev = [
        stats.t.std(d, loc=m, scale=s) for (d, m, s) in zip(df, mu, sigma)
    ]
    self.assertAllClose(expected_stddev, stddev)

  def testMode(self):
    # [FUNCTION DOC]
    # Test: for degrees of freedom 0.5, 1, 3, asserts the mode equals the location values
    # (-1, 0, 1) regardless of degrees of freedom or scale.
    df = [0.5, 1., 3]
    mu = [-1, 0., 1]
    sigma = [5., 4., 3.]
    student = student_t.StudentT(df=df, loc=mu, scale=sigma)
    # Test broadcast of mu across shape of df/sigma
    mode = self.evaluate(student.mode())
    self.assertAllClose([-1., 0, 1], mode)

  def testPdfOfSample(self):
    # [FUNCTION DOC]
    # Test: draws 20000 seeded samples from a scalar distribution (degrees of freedom 3,
    # location pi, scale 1), evaluates the density at the samples and at the mean, and asserts
    # static shapes, sample mean near pi (0.1), mean equal to pi, the trapezoid-integrated
    # density over the samples near one (loose 0.05 tolerance), and with scipy the density at
    # the mean matching scipy's.
    student = student_t.StudentT(df=3., loc=np.pi, scale=1.)
    num = 20000
    samples = student.sample(num, seed=123456)
    pdfs = student.prob(samples)
    mean = student.mean()
    mean_pdf = student.prob(student.mean())
    sample_vals, pdf_vals, mean_val, mean_pdf_val = self.evaluate(
        [samples, pdfs, student.mean(), mean_pdf])
    self.assertEqual(samples.get_shape(), (num,))
    self.assertEqual(pdfs.get_shape(), (num,))
    self.assertEqual(mean.get_shape(), ())
    self.assertNear(np.pi, np.mean(sample_vals), err=0.1)
    self.assertNear(np.pi, mean_val, err=1e-6)
    # Verify integral over sample*pdf ~= 1.
    # Tolerance increased since eager was getting a value of 1.002041.
    self._assertIntegral(sample_vals, pdf_vals, err=5e-2)
    if not stats:
      return
    self.assertNear(stats.t.pdf(np.pi, 3., loc=np.pi), mean_pdf_val, err=1e-6)

  def testFullyReparameterized(self):
    # [FUNCTION DOC]
    # Test: records, with a gradient tape watching degrees of freedom, location and scale, the
    # drawing of 100 samples and asserts the gradient of the samples with respect to each of the
    # three parameters is not null, i.e. sampling is fully reparameterized.
    df = constant_op.constant(2.0)
    mu = constant_op.constant(1.0)
    sigma = constant_op.constant(3.0)
    with backprop.GradientTape() as tape:
      tape.watch(df)
      tape.watch(mu)
      tape.watch(sigma)
      student = student_t.StudentT(df=df, loc=mu, scale=sigma)
      samples = student.sample(100)
    grad_df, grad_mu, grad_sigma = tape.gradient(samples, [df, mu, sigma])
    self.assertIsNotNone(grad_df)
    self.assertIsNotNone(grad_mu)
    self.assertIsNotNone(grad_sigma)

  def testPdfOfSampleMultiDims(self):
    # [FUNCTION DOC]
    # Test: distribution with batch shape two by two (degrees of freedom 7 and 11, locations 5
    # and 6, scale 3). Asserts static and dynamic event shape is empty and batch shape is two by
    # two; draws 50000 samples, asserts sample and density shapes, per-row sample means near 5
    # and 6, each batch member's integrated density near one (0.05), and with scipy that the
    # sample variance per degrees-of-freedom column is within 1.0 of scipy's variance.
    student = student_t.StudentT(df=[7., 11.], loc=[[5.], [6.]], scale=3.)
    self.assertAllEqual([], student.event_shape)
    self.assertAllEqual([], self.evaluate(student.event_shape_tensor()))
    self.assertAllEqual([2, 2], student.batch_shape)
    self.assertAllEqual([2, 2], self.evaluate(student.batch_shape_tensor()))
    num = 50000
    samples = student.sample(num, seed=123456)
    pdfs = student.prob(samples)
    sample_vals, pdf_vals = self.evaluate([samples, pdfs])
    self.assertEqual(samples.get_shape(), (num, 2, 2))
    self.assertEqual(pdfs.get_shape(), (num, 2, 2))
    self.assertNear(5., np.mean(sample_vals[:, 0, :]), err=0.1)
    self.assertNear(6., np.mean(sample_vals[:, 1, :]), err=0.1)
    self._assertIntegral(sample_vals[:, 0, 0], pdf_vals[:, 0, 0], err=0.05)
    self._assertIntegral(sample_vals[:, 0, 1], pdf_vals[:, 0, 1], err=0.05)
    self._assertIntegral(sample_vals[:, 1, 0], pdf_vals[:, 1, 0], err=0.05)
    self._assertIntegral(sample_vals[:, 1, 1], pdf_vals[:, 1, 1], err=0.05)
    if not stats:
      return
    self.assertNear(
        stats.t.var(7., loc=0., scale=3.),  # loc d.n. effect var
        np.var(sample_vals[:, :, 0]),
        err=1.0)
    self.assertNear(
        stats.t.var(11., loc=0., scale=3.),  # loc d.n. effect var
        np.var(sample_vals[:, :, 1]),
        err=1.0)

  def _assertIntegral(self, sample_vals, pdf_vals, err=1.5e-3):
    # [FUNCTION DOC]
    # HELPER: numerical-integration assertion for the density tests.
    # Inputs: one-dimensional arrays of sample points and the density values at them, and an
    # absolute error tolerance (default 0.0015). Sorts the points, integrates the density with the
    # trapezoid rule starting from a point 1000 below the smallest sample (with density zero
    # there), and asserts the total is within the tolerance of one. Returns nothing.
    s_p = zip(sample_vals, pdf_vals)
    prev = (sample_vals.min() - 1000, 0)
    total = 0
    for k in sorted(s_p, key=lambda x: x[0]):
      pair_pdf = (k[1] + prev[1]) / 2
      total += (k[0] - prev[0]) * pair_pdf
      prev = k
    self.assertNear(1., total, err=err)

  def testNegativeDofFails(self):
    # [FUNCTION DOC]
    # Test: constructing a distribution with argument validation enabled and a negative degrees
    # of freedom entry (-5) and then evaluating its mean must raise an op error saying the
    # positivity condition did not hold.
    with self.assertRaisesOpError(r"Condition x > 0 did not hold"):
      student = student_t.StudentT(
          df=[2, -5.], loc=0., scale=1., validate_args=True, name="S")
      self.evaluate(student.mean())

  def testStudentTWithAbsDfSoftplusScale(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name and the test's use of absolute value suggest degrees of freedom are
    # transformed by absolute value, but the assertion compares against the floor of the
    # absolute value (-3.2 becomes 3, not 3.2), i.e. the transform also truncates.
    # Test: builds the variant class from negative degrees of freedom, locations and negative
    # scales, and asserts its degrees of freedom equal floor of the absolute input, its location
    # is unchanged, and its scale equals the softplus of the input scale.
    df = constant_op.constant([-3.2, -4.6])
    mu = constant_op.constant([-4.2, 3.4])
    sigma = constant_op.constant([-6.4, -8.8])
    student = student_t.StudentTWithAbsDfSoftplusScale(
        df=df, loc=mu, scale=sigma)
    self.assertAllClose(
        math_ops.floor(self.evaluate(math_ops.abs(df))),
        self.evaluate(student.df))
    self.assertAllClose(self.evaluate(mu), self.evaluate(student.loc))
    self.assertAllClose(
        self.evaluate(nn_ops.softplus(sigma)), self.evaluate(student.scale))


if __name__ == "__main__":
  test.main()
