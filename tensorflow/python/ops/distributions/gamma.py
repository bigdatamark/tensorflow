# [MODULE DOC]
# Defines the Gamma distribution (legacy tf.distributions API, v1 only) over positive reals,
# parameterized by concentration and rate, with reparameterized sampling via the random gamma op.
# Also defines GammaWithSoftplusConcentrationRate, a convenience subclass that passes its raw
# parameters through softplus, and registers the closed-form KL divergence between two Gamma
# distributions. Import-time side effects: exports Gamma under the v1 distributions namespace and
# registers the Gamma-Gamma KL function with the kullback_leibler registry.
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
"""The Gamma distribution class."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


__all__ = [
    "Gamma",
    "GammaWithSoftplusConcentrationRate",
]


@tf_export(v1=["distributions.Gamma"])
class Gamma(distribution.Distribution):
  # [CLASS DOC]
  # Gamma distribution over positive real numbers, parameterized by a concentration (shape) and
  # a rate (inverse scale), both positive tensors that broadcast against each other. Holds the two
  # parameters as identity-wrapped tensors; the batch shape is their broadcast shape and the event
  # shape is scalar. Samples are fully reparameterized (implicit reparameterization gradients).
  # Supports pdf/cdf, entropy, mean, variance, stddev and mode. Created directly by the caller.
  # Deprecated in favor of the TensorFlow Probability distributions library.
  """Gamma distribution.

  The Gamma distribution is defined over positive real numbers using
  parameters `concentration` (aka "alpha") and `rate` (aka "beta").

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, beta, x > 0) = x**(alpha - 1) exp(-x beta) / Z
  Z = Gamma(alpha) beta**(-alpha)
  ```

  where:

  * `concentration = alpha`, `alpha > 0`,
  * `rate = beta`, `beta > 0`,
  * `Z` is the normalizing constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The cumulative density function (cdf) is,

  ```none
  cdf(x; alpha, beta, x > 0) = GammaInc(alpha, beta x) / Gamma(alpha)
  ```

  where `GammaInc` is the [lower incomplete Gamma function](
  https://en.wikipedia.org/wiki/Incomplete_gamma_function).

  The parameters can be intuited via their relationship to mean and stddev,

  ```none
  concentration = alpha = (mean / stddev)**2
  rate = beta = mean / stddev**2 = concentration / mean
  ```

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  Warning: The samples of this distribution are always non-negative. However,
  the samples that are smaller than `np.finfo(dtype).tiny` are rounded
  to this value, so it appears more often than it should.
  This should only be noticeable when the `concentration` is very small, or the
  `rate` is very large. See note in `tf.random.gamma` docstring.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in
  (Figurnov et al., 2018).

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dist = tfd.Gamma(concentration=3.0, rate=2.0)
  dist2 = tfd.Gamma(concentration=[3.0, 4.0], rate=[2.0, 3.0])
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  concentration = tf.constant(3.0)
  rate = tf.constant(2.0)
  dist = tfd.Gamma(concentration, rate)
  samples = dist.sample(5)  # Shape [5]
  loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tf.gradients(loss, [concentration, rate])
  ```

  References:
    Implicit Reparameterization Gradients:
      [Figurnov et al., 2018]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
      ([pdf](http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
  """

  @deprecation.deprecated(
      "2019-01-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.distributions`.",
      warn_once=True)
  def __init__(self,
               concentration,
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name="Gamma"):
    # [FUNCTION DOC]
    # Constructor. Inputs: concentration and rate (floating point tensors or values, must be
    # positive and broadcast together), validate_args (bool, default false; when true adds
    # assertions that both are strictly positive), allow_nan_stats (bool, default true; whether
    # undefined statistics return NaN instead of raising), name (string scope name).
    # Stores identity copies of both parameters, checks they share one floating dtype, and
    # initializes the base distribution with fully reparameterized sampling and the original
    # constructor arguments recorded as parameters. Raises a type error if the dtypes differ.
    # Wrapped by a deprecation decorator that warns once on first use.
    """Construct Gamma with `concentration` and `rate` parameters.

    The parameters `concentration` and `rate` must be shaped in a way that
    supports broadcasting (e.g. `concentration + rate` is a valid operation).

    Args:
      concentration: Floating point tensor, the concentration params of the
        distribution(s). Must contain only positive values.
      rate: Floating point tensor, the inverse scale params of the
        distribution(s). Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `concentration` and `rate` are different dtypes.
    """
    parameters = dict(locals())
    with ops.name_scope(name, values=[concentration, rate]) as name:
      with ops.control_dependencies([
          check_ops.assert_positive(concentration),
          check_ops.assert_positive(rate),
      ] if validate_args else []):
        self._concentration = array_ops.identity(
            concentration, name="concentration")
        self._rate = array_ops.identity(rate, name="rate")
        check_ops.assert_same_float_dtype(
            [self._concentration, self._rate])
    super(Gamma, self).__init__(
        dtype=self._concentration.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=distribution.FULLY_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[self._concentration,
                       self._rate],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    # [FUNCTION DOC]
    # HELPER: static helper used by the base distribution's param_shapes machinery.
    # Given a sample shape (shape tensor or list), returns a mapping from the parameter names
    # concentration and rate to the same shape converted to a 32-bit integer tensor. Both
    # parameters share the sample shape since they are scalar per batch member.
    return dict(
        zip(("concentration", "rate"), ([ops.convert_to_tensor(
            sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def concentration(self):
    # [FUNCTION DOC]
    # Property returning the concentration (shape, alpha) parameter tensor given at construction.
    """Concentration parameter."""
    return self._concentration

  @property
  def rate(self):
    # [FUNCTION DOC]
    # Property returning the rate (inverse scale, beta) parameter tensor given at construction.
    """Rate parameter."""
    return self._rate

  def _batch_shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: hook used by the base class batch_shape_tensor.
    # Returns the dynamic batch shape as an integer tensor: the broadcast of the dynamic shapes
    # of concentration and rate. Takes no arguments.
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.concentration),
        array_ops.shape(self.rate))

  def _batch_shape(self):
    # [FUNCTION DOC]
    # HELPER: hook used by the base class batch_shape property.
    # Returns the static batch shape: the broadcast of the static shapes of concentration and rate.
    # Raises a value error if those shapes are not broadcast compatible.
    return array_ops.broadcast_static_shape(
        self.concentration.get_shape(),
        self.rate.get_shape())

  def _event_shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: hook used by the base class event_shape_tensor.
    # Returns an empty 32-bit integer vector constant, since each Gamma draw is a scalar.
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    # [FUNCTION DOC]
    # HELPER: hook used by the base class event_shape property.
    # Returns the static event shape, an empty (scalar) tensor shape.
    return tensor_shape.TensorShape([])

  @distribution_util.AppendDocstring(
      """Note: See `tf.random.gamma` docstring for sampling details and
      caveats.""")
  def _sample_n(self, n, seed=None):
    # [FUNCTION DOC]
    # HELPER: hook used by the base class sample method.
    # Draws n independent samples per batch member using the random gamma op with the
    # concentration as alpha and the rate as beta. n is an integer count, seed is an optional
    # integer random seed (default none). Returns a tensor of the distribution dtype whose shape
    # is n followed by the batch shape. Consumes randomness; the docstring is appended to the
    # public sample docs.
    return random_ops.random_gamma(
        shape=[n],
        alpha=self.concentration,
        beta=self.rate,
        dtype=self.dtype,
        seed=seed)

  def _log_prob(self, x):
    # [FUNCTION DOC]
    # HELPER: hook used by the base class log_prob.
    # Returns the log density at x (tensor of positive values) as the unnormalized log
    # probability minus the log normalization constant. Validates x when validate_args is set.
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _cdf(self, x):
    # [FUNCTION DOC]
    # HELPER: hook used by the base class cdf.
    # Returns the cumulative distribution at x (tensor of positive values), computed as the
    # regularized lower incomplete gamma function of the concentration and rate times x.
    # Validates x first (positivity checked only when validate_args is set; dtype always checked).
    x = self._maybe_assert_valid_sample(x)
    # Note that igamma returns the regularized incomplete gamma function,
    # which is what we want for the CDF.
    return math_ops.igamma(self.concentration, self.rate * x)

  def _log_unnormalized_prob(self, x):
    # [FUNCTION DOC]
    # HELPER: part of the log density computation.
    # Returns (concentration - 1) times log x (using the safe x-log-y op so zero is handled)
    # minus rate times x, for x a tensor of positive values. Validates x first.
    x = self._maybe_assert_valid_sample(x)
    return math_ops.xlogy(self.concentration - 1., x) - self.rate * x

  def _log_normalization(self):
    # [FUNCTION DOC]
    # HELPER: part of the log density computation.
    # Returns the log of the normalizing constant: log-gamma of concentration minus concentration
    # times log rate. Takes no arguments; returns a tensor with the batch shape.
    return (math_ops.lgamma(self.concentration)
            - self.concentration * math_ops.log(self.rate))

  def _entropy(self):
    # [FUNCTION DOC]
    # HELPER: hook used by the base class entropy.
    # Returns the differential entropy per batch member: concentration minus log rate plus
    # log-gamma of concentration plus (1 - concentration) times digamma of concentration.
    return (self.concentration
            - math_ops.log(self.rate)
            + math_ops.lgamma(self.concentration)
            + ((1. - self.concentration) *
               math_ops.digamma(self.concentration)))

  def _mean(self):
    # [FUNCTION DOC]
    # HELPER: hook used by the base class mean.
    # Returns the mean, concentration divided by rate (broadcast over the batch shape).
    return self.concentration / self.rate

  def _variance(self):
    # [FUNCTION DOC]
    # HELPER: hook used by the base class variance.
    # Returns the variance, concentration divided by the square of rate.
    return self.concentration / math_ops.square(self.rate)

  def _stddev(self):
    # [FUNCTION DOC]
    # HELPER: hook used by the base class stddev.
    # Returns the standard deviation, square root of concentration divided by rate.
    return math_ops.sqrt(self.concentration) / self.rate

  @distribution_util.AppendDocstring(
      """The mode of a gamma distribution is `(shape - 1) / rate` when
      `shape > 1`, and `NaN` otherwise. If `self.allow_nan_stats` is `False`,
      an exception will be raised rather than returning `NaN`.""")
  def _mode(self):
    # [FUNCTION DOC]
    # HELPER: hook used by the base class mode.
    # KNOWN ISSUE: the appended docstring talks about `shape` but the parameter is called
    # concentration; also it says the mode is undefined for shape <= 1 while the code computes
    # the formula unconditionally and only masks it with NaN when allow_nan_stats is true.
    # Returns (concentration - 1) / rate. If allow_nan_stats is true, entries whose concentration
    # is not greater than 1 are replaced by NaN. Otherwise the result is made to depend on an
    # assertion that 1 is less than concentration, which fails at run time if any is not.
    mode = (self.concentration - 1.) / self.rate
    if self.allow_nan_stats:
      nan = array_ops.fill(
          self.batch_shape_tensor(),
          np.array(np.nan, dtype=self.dtype.as_numpy_dtype()),
          name="nan")
      return array_ops.where_v2(self.concentration > 1., mode, nan)
    else:
      return control_flow_ops.with_dependencies([
          check_ops.assert_less(
              array_ops.ones([], self.dtype),
              self.concentration,
              message="mode not defined when any concentration <= 1"),
          ], mode)

  def _maybe_assert_valid_sample(self, x):
    # [FUNCTION DOC]
    # HELPER: input validation for cdf and log density computations.
    # Checks that x (tensor) has the distribution floating dtype (raises a type error otherwise).
    # If validate_args is false returns x unchanged; otherwise returns x made dependent on an
    # assertion that all entries are strictly positive.
    check_ops.assert_same_float_dtype(tensors=[x], dtype=self.dtype)
    if not self.validate_args:
      return x
    return control_flow_ops.with_dependencies([
        check_ops.assert_positive(x),
    ], x)


class GammaWithSoftplusConcentrationRate(Gamma):
  # [CLASS DOC]
  # Gamma distribution whose constructor applies softplus to the supplied concentration and rate
  # so that unconstrained real values can be used. Otherwise identical to Gamma. Deprecated.
  """`Gamma` with softplus of `concentration` and `rate`."""

  @deprecation.deprecated(
      "2019-01-01",
      "Use `tfd.Gamma(tf.nn.softplus(concentration), "
      "tf.nn.softplus(rate))` instead.",
      warn_once=True)
  def __init__(self,
               concentration,
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name="GammaWithSoftplusConcentrationRate"):
    # [FUNCTION DOC]
    # Constructor taking the same arguments as Gamma (concentration, rate, validate_args,
    # allow_nan_stats, name). Applies softplus to concentration and rate inside a name scope,
    # then calls the Gamma constructor with the transformed values. Afterwards it overwrites the
    # stored parameters with the raw arguments originally passed (not the softplus outputs).
    # Deprecated; warns once.
    parameters = dict(locals())
    with ops.name_scope(name, values=[concentration, rate]) as name:
      super(GammaWithSoftplusConcentrationRate, self).__init__(
          concentration=nn.softplus(concentration,
                                    name="softplus_concentration"),
          rate=nn.softplus(rate, name="softplus_rate"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters


@kullback_leibler.RegisterKL(Gamma, Gamma)
def _kl_gamma_gamma(g0, g1, name=None):
  # [FUNCTION DOC]
  # HELPER: registered KL implementation for the Gamma-Gamma pair.
  # Computes the batched closed-form KL divergence from g0 to g1, both Gamma distributions, from
  # their concentrations and rates using digamma, log-gamma and log terms. name is an optional
  # scope name (default kl_gamma_gamma). Returns a tensor with the broadcast batch shape.
  """Calculate the batched KL divergence KL(g0 || g1) with g0 and g1 Gamma.

  Args:
    g0: instance of a Gamma distribution object.
    g1: instance of a Gamma distribution object.
    name: (optional) Name to use for created operations.
      Default is "kl_gamma_gamma".

  Returns:
    kl_gamma_gamma: `Tensor`. The batchwise KL(g0 || g1).
  """
  with ops.name_scope(name, "kl_gamma_gamma", values=[
      g0.concentration, g0.rate, g1.concentration, g1.rate]):
    # Result from:
    #   http://www.fil.ion.ucl.ac.uk/~wpenny/publications/densities.ps
    # For derivation see:
    #   http://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions   pylint: disable=line-too-long
    return (((g0.concentration - g1.concentration)
             * math_ops.digamma(g0.concentration))
            + math_ops.lgamma(g1.concentration)
            - math_ops.lgamma(g0.concentration)
            + g1.concentration * math_ops.log(g0.rate)
            - g1.concentration * math_ops.log(g1.rate)
            + g0.concentration * (g1.rate / g0.rate - 1.))
