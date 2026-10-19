# [MODULE DOC]
# Unit tests for the Uniform distribution class in the distributions ops package: endpoints and
# range, density and log density (including broadcasting and values outside the support), static
# and dynamic shapes, cumulative functions, entropy, argument validation, sampling, moments
# (compared with scipy when importable), NaN propagation, reparameterized gradients and float64
# support. Import-time side effect: attempts the optional scipy import and logs a warning when
# it fails. Runs as a test binary through the TensorFlow test main.
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
"""Tests for Uniform distribution."""

import importlib

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import uniform as uniform_lib
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


class UniformTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the Uniform distribution. Most tests run in both graph and eager modes
  # through per-method decorators; two (the reparameterization test and the float64 test) have
  # no such decorator and use the default mode. No shared fixtures: each test builds its own
  # distribution.

  @test_util.run_in_graph_and_eager_modes
  def testUniformRange(self):
    # [FUNCTION DOC]
    # Test: creates a scalar uniform distribution on 3 to 10 and asserts the low and high
    # properties equal the inputs and the range method returns their difference (7).
    a = 3.0
    b = 10.0
    uniform = uniform_lib.Uniform(low=a, high=b)
    self.assertAllClose(a, self.evaluate(uniform.low))
    self.assertAllClose(b, self.evaluate(uniform.high))
    self.assertAllClose(b - a, self.evaluate(uniform.range()))

  @test_util.run_in_graph_and_eager_modes
  def testUniformPDF(self):
    # [FUNCTION DOC]
    # Test: batch of six with five members on -3 to 11 and one on 15 to 20, evaluated at six
    # points (some outside the support). Asserts density equals one over the width inside the
    # support and zero outside (the last point is inside the last member), computed by a nested
    # helper, and that the log density equals the log of that expectation.
    a = constant_op.constant([-3.0] * 5 + [15.0])
    b = constant_op.constant([11.0] * 5 + [20.0])
    uniform = uniform_lib.Uniform(low=a, high=b)

    a_v = -3.0
    b_v = 11.0
    x = np.array([-10.5, 4.0, 0.0, 10.99, 11.3, 17.0], dtype=np.float32)

    def _expected_pdf():
      # [FUNCTION DOC]
      # HELPER: nested oracle for the density test. Takes no arguments; uses the enclosing
      # test's evaluation points and bounds. Returns an array with one over the width of the
      # first support everywhere, zero for points outside it, and the value for the last
      # (separate) batch member hard-coded as one over five.
      pdf = np.zeros_like(x) + 1.0 / (b_v - a_v)
      pdf[x > b_v] = 0.0
      pdf[x < a_v] = 0.0
      pdf[5] = 1.0 / (20.0 - 15.0)
      return pdf

    expected_pdf = _expected_pdf()

    pdf = uniform.prob(x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

    log_pdf = uniform.log_prob(x)
    self.assertAllClose(np.log(expected_pdf), self.evaluate(log_pdf))

  @test_util.run_in_graph_and_eager_modes
  def testUniformShape(self):
    # [FUNCTION DOC]
    # Test: with a length-five low and scalar high, asserts the dynamic and static batch shape
    # are (5) and the dynamic and static event shapes are empty.
    a = constant_op.constant([-3.0] * 5)
    b = constant_op.constant(11.0)
    uniform = uniform_lib.Uniform(low=a, high=b)

    self.assertEqual(self.evaluate(uniform.batch_shape_tensor()), (5,))
    self.assertEqual(uniform.batch_shape, tensor_shape.TensorShape([5]))
    self.assertAllEqual(self.evaluate(uniform.event_shape_tensor()), [])
    self.assertEqual(uniform.event_shape, tensor_shape.TensorShape([]))

  @test_util.run_in_graph_and_eager_modes
  def testUniformPDFWithScalarEndpoint(self):
    # [FUNCTION DOC]
    # Test: low is a vector (0, 5) and high is a scalar 10; density at the points 0 and 8 must
    # equal one over each member's width, verifying scalar endpoint broadcasting.
    a = constant_op.constant([0.0, 5.0])
    b = constant_op.constant(10.0)
    uniform = uniform_lib.Uniform(low=a, high=b)

    x = np.array([0.0, 8.0], dtype=np.float32)
    expected_pdf = np.array([1.0 / (10.0 - 0.0), 1.0 / (10.0 - 5.0)])

    pdf = uniform.prob(x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

  @test_util.run_in_graph_and_eager_modes
  def testUniformCDF(self):
    # [FUNCTION DOC]
    # Test: batch of six on 1 to 11 evaluated at six points including ones below and above the
    # support. Asserts the CDF equals the linear ramp clipped to zero below and one at or above
    # the upper end (computed by a nested helper), and the log CDF equals its log (negative
    # infinity where zero).
    batch_size = 6
    a = constant_op.constant([1.0] * batch_size)
    b = constant_op.constant([11.0] * batch_size)
    a_v = 1.0
    b_v = 11.0
    x = np.array([-2.5, 2.5, 4.0, 0.0, 10.99, 12.0], dtype=np.float32)

    uniform = uniform_lib.Uniform(low=a, high=b)

    def _expected_cdf():
      # [FUNCTION DOC]
      # HELPER: nested oracle for the CDF test. Takes no arguments; uses the enclosing test's
      # points and bounds. Returns the linear ramp, set to one for points at or above the upper
      # bound and zero for points below the lower bound. Mutates its own local array only.
      cdf = (x - a_v) / (b_v - a_v)
      cdf[x >= b_v] = 1
      cdf[x < a_v] = 0
      return cdf

    cdf = uniform.cdf(x)
    self.assertAllClose(_expected_cdf(), self.evaluate(cdf))

    log_cdf = uniform.log_cdf(x)
    self.assertAllClose(np.log(_expected_cdf()), self.evaluate(log_cdf))

  @test_util.run_in_graph_and_eager_modes
  def testUniformEntropy(self):
    # [FUNCTION DOC]
    # Test: low is a length-three vector and high a one-by-three row, broadcasting to one by
    # three; asserts entropy equals the log of the width.
    a_v = np.array([1.0, 1.0, 1.0])
    b_v = np.array([[1.5, 2.0, 3.0]])
    uniform = uniform_lib.Uniform(low=a_v, high=b_v)

    expected_entropy = np.log(b_v - a_v)
    self.assertAllClose(expected_entropy, self.evaluate(uniform.entropy()))

  @test_util.run_in_graph_and_eager_modes
  def testUniformAssertMaxGtMin(self):
    # [FUNCTION DOC]
    # Test: constructs the distribution with validation enabled where the first member has equal
    # low and high; evaluating the low property must raise an invalid-argument error whose
    # message contains the failed "x < y" assertion.
    a_v = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    b_v = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                             "x < y"):
      uniform = uniform_lib.Uniform(low=a_v, high=b_v, validate_args=True)
      self.evaluate(uniform.low)

  @test_util.run_in_graph_and_eager_modes
  def testUniformSample(self):
    # [FUNCTION DOC]
    # Test: low is (3, 4), high is scalar 13; draws 100000 seeded samples and asserts shape
    # 100000 by 2, each column's mean within 0.1 of the midpoint, and no sample below its lower
    # bound or at/above the upper bound.
    a = constant_op.constant([3.0, 4.0])
    b = constant_op.constant(13.0)
    a1_v = 3.0
    a2_v = 4.0
    b_v = 13.0
    n = constant_op.constant(100000)
    uniform = uniform_lib.Uniform(low=a, high=b)

    samples = uniform.sample(n, seed=137)
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (100000, 2))
    self.assertAllClose(
        sample_values[::, 0].mean(), (b_v + a1_v) / 2, atol=1e-1, rtol=0.)
    self.assertAllClose(
        sample_values[::, 1].mean(), (b_v + a2_v) / 2, atol=1e-1, rtol=0.)
    self.assertFalse(
        np.any(sample_values[::, 0] < a1_v) or np.any(sample_values >= b_v))
    self.assertFalse(
        np.any(sample_values[::, 1] < a2_v) or np.any(sample_values >= b_v))

  @test_util.run_in_graph_and_eager_modes
  def _testUniformSampleMultiDimensional(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this test is disabled. The leading underscore keeps the test runner from
    # discovering it, so it never runs (the in-code note says to enable it once an external bug
    # is fixed).
    # Intended scenario: two-by-two batch (low 3 and 22, high 13 and 35); draws 100000 samples,
    # asserts static shape, that first-row samples stay within their bounds and that their means
    # are within 0.01 of the midpoints.
    # DISABLED: Please enable this test once b/issues/30149644 is resolved.
    batch_size = 2
    a_v = [3.0, 22.0]
    b_v = [13.0, 35.0]
    a = constant_op.constant([a_v] * batch_size)
    b = constant_op.constant([b_v] * batch_size)

    uniform = uniform_lib.Uniform(low=a, high=b)

    n_v = 100000
    n = constant_op.constant(n_v)
    samples = uniform.sample(n)
    self.assertEqual(samples.get_shape(), (n_v, batch_size, 2))

    sample_values = self.evaluate(samples)

    self.assertFalse(
        np.any(sample_values[:, 0, 0] < a_v[0]) or
        np.any(sample_values[:, 0, 0] >= b_v[0]))
    self.assertFalse(
        np.any(sample_values[:, 0, 1] < a_v[1]) or
        np.any(sample_values[:, 0, 1] >= b_v[1]))

    self.assertAllClose(
        sample_values[:, 0, 0].mean(), (a_v[0] + b_v[0]) / 2, atol=1e-2)
    self.assertAllClose(
        sample_values[:, 0, 1].mean(), (a_v[1] + b_v[1]) / 2, atol=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def testUniformMean(self):
    # [FUNCTION DOC]
    # Test: uniform on 10 to 100; asserts the mean equals scipy's uniform mean. Returns without
    # asserting when scipy is unavailable.
    a = 10.0
    b = 100.0
    uniform = uniform_lib.Uniform(low=a, high=b)
    if not stats:
      return
    s_uniform = stats.uniform(loc=a, scale=b - a)
    self.assertAllClose(self.evaluate(uniform.mean()), s_uniform.mean())

  @test_util.run_in_graph_and_eager_modes
  def testUniformVariance(self):
    # [FUNCTION DOC]
    # Test: uniform on 10 to 100; asserts the variance equals scipy's. Returns without
    # asserting when scipy is unavailable.
    a = 10.0
    b = 100.0
    uniform = uniform_lib.Uniform(low=a, high=b)
    if not stats:
      return
    s_uniform = stats.uniform(loc=a, scale=b - a)
    self.assertAllClose(self.evaluate(uniform.variance()), s_uniform.var())

  @test_util.run_in_graph_and_eager_modes
  def testUniformStd(self):
    # [FUNCTION DOC]
    # Test: uniform on 10 to 100; asserts the standard deviation equals scipy's. Returns without
    # asserting when scipy is unavailable.
    a = 10.0
    b = 100.0
    uniform = uniform_lib.Uniform(low=a, high=b)
    if not stats:
      return
    s_uniform = stats.uniform(loc=a, scale=b - a)
    self.assertAllClose(self.evaluate(uniform.stddev()), s_uniform.std())

  @test_util.run_in_graph_and_eager_modes
  def testUniformNans(self):
    # [FUNCTION DOC]
    # Test: builds a stacked input of one regular value and one NaN (made by dividing zero by
    # zero) and asserts the density is not NaN for the regular value and NaN for the NaN input,
    # i.e. NaN propagates through the density.
    a = 10.0
    b = [11.0, 100.0]
    uniform = uniform_lib.Uniform(low=a, high=b)

    no_nans = constant_op.constant(1.0)
    nans = constant_op.constant(0.0) / constant_op.constant(0.0)
    self.assertTrue(self.evaluate(math_ops.is_nan(nans)))
    with_nans = array_ops_stack.stack([no_nans, nans])

    pdf = uniform.prob(with_nans)

    is_nan = self.evaluate(math_ops.is_nan(pdf))
    self.assertFalse(is_nan[0])
    self.assertTrue(is_nan[1])

  @test_util.run_in_graph_and_eager_modes
  def testUniformSamplePdf(self):
    # [FUNCTION DOC]
    # Test: draws 10 samples from a two-member distribution (positional low and high) and
    # asserts that the density at every sample is strictly positive.
    a = 10.0
    b = [11.0, 100.0]
    uniform = uniform_lib.Uniform(a, b)
    self.assertTrue(
        self.evaluate(
            math_ops.reduce_all(uniform.prob(uniform.sample(10)) > 0)))

  @test_util.run_in_graph_and_eager_modes
  def testUniformBroadcasting(self):
    # [FUNCTION DOC]
    # Test: scalar low 10, high (11, 20); density of a three-by-two grid of points must be the
    # expected matrix of one, 0.1 and zero, verifying broadcasting of points against the batch.
    a = 10.0
    b = [11.0, 20.0]
    uniform = uniform_lib.Uniform(a, b)

    pdf = uniform.prob([[10.5, 11.5], [9.0, 19.0], [10.5, 21.0]])
    expected_pdf = np.array([[1.0, 0.1], [0.0, 0.1], [1.0, 0.0]])
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

  @test_util.run_in_graph_and_eager_modes
  def testUniformSampleWithShape(self):
    # [FUNCTION DOC]
    # Test: same two-member distribution; the density at samples drawn with sample shape (2, 3)
    # must equal one and 0.1 for the two members across a two-by-three-by-two array, and the
    # density at a sample drawn without shape must equal (1, 0.1). Verifies the samples lie in the
    # support.
    a = 10.0
    b = [11.0, 20.0]
    uniform = uniform_lib.Uniform(a, b)

    pdf = uniform.prob(uniform.sample((2, 3)))
    # pylint: disable=bad-continuation
    expected_pdf = [
        [[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]],
        [[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]],
    ]
    # pylint: enable=bad-continuation
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

    pdf = uniform.prob(uniform.sample())
    expected_pdf = [1.0, 0.1]
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

  def testFullyReparameterized(self):
    # [FUNCTION DOC]
    # Test: records, with a gradient tape watching the low and high tensors, the drawing of 100
    # samples and asserts the gradient with respect to each endpoint is not null (sampling is
    # reparameterized).
    a = constant_op.constant(0.1)
    b = constant_op.constant(0.8)
    with backprop.GradientTape() as tape:
      tape.watch(a)
      tape.watch(b)
      uniform = uniform_lib.Uniform(a, b)
      samples = uniform.sample(100)
    grad_a, grad_b = tape.gradient(samples, [a, b])
    self.assertIsNotNone(grad_a)
    self.assertIsNotNone(grad_b)

  # Eager doesn't pass due to a type mismatch in one of the ops.
  def testUniformFloat64(self):
    # [FUNCTION DOC]
    # Test: uniform on 0 to 1 built from double-precision scalars. Asserts density at 0.5 and
    # 0.6 is one, CDF equals the input points, mean is 0.5, variance is one twelfth and entropy
    # is zero. Per the comment above it, eager mode is not covered because of a type mismatch in
    # one operation, so no mode decorator is applied.
    uniform = uniform_lib.Uniform(
        low=np.float64(0.), high=np.float64(1.))

    self.assertAllClose(
        [1., 1.],
        self.evaluate(uniform.prob(np.array([0.5, 0.6], dtype=np.float64))))

    self.assertAllClose(
        [0.5, 0.6],
        self.evaluate(uniform.cdf(np.array([0.5, 0.6], dtype=np.float64))))

    self.assertAllClose(0.5, self.evaluate(uniform.mean()))
    self.assertAllClose(1 / 12., self.evaluate(uniform.variance()))
    self.assertAllClose(0., self.evaluate(uniform.entropy()))


if __name__ == "__main__":
  test.main()
