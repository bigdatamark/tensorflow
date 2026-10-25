# [MODULE DOC]
# Defines the deprecated (v1 only) Dirichlet distribution over the probability simplex, plus the
# registered closed-form KL divergence between two Dirichlet distributions. Exposes the class
# Dirichlet (exported under the v1 distributions namespace) via the module-level export list.
# Builds on the generic Distribution base class and the KL registry in sibling modules.
# Import-time side effects: registers the public API symbol with the export decorator and
# registers the Dirichlet-to-Dirichlet KL function with the KL registry. Also holds a shared
# documentation note appended to the probability-method docstrings.
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
"""The Dirichlet distribution class."""

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


__all__ = [
    "Dirichlet",
]


_dirichlet_sample_note = """Note: `value` must be a non-negative tensor with
dtype `self.dtype` and be in the `(self.event_shape() - 1)`-simplex, i.e.,
`tf.reduce_sum(value, -1) = 1`. It must have a shape compatible with
`self.batch_shape() + self.event_shape()`."""


@tf_export(v1=["distributions.Dirichlet"])
class Dirichlet(distribution.Distribution):
  # [CLASS DOC]
  # Dirichlet distribution over the open simplex, parameterized by a positive concentration
  # tensor whose last dimension is the event size (at least 2); leading dimensions form the
  # batch shape. Holds the concentration tensor and its sum over the last axis (total
  # concentration). Subclasses the generic Distribution, reports fully reparameterized samples
  # (gamma-based sampling with implicit gradients), and implements the private hooks (shapes,
  # sampling, log/prob, entropy, mean, mode, covariance, variance) that the base class public
  # methods call. Created by passing a concentration array-like; deprecated in favor of the
  # separate probability library.
  """Dirichlet distribution.

  The Dirichlet distribution is defined over the
  [`(k-1)`-simplex](https://en.wikipedia.org/wiki/Simplex) using a positive,
  length-`k` vector `concentration` (`k > 1`). The Dirichlet is identically the
  Beta distribution when `k = 2`.

  #### Mathematical Details

  The Dirichlet is a distribution over the open `(k-1)`-simplex, i.e.,

  ```none
  S^{k-1} = { (x_0, ..., x_{k-1}) in R^k : sum_j x_j = 1 and all_j x_j > 0 }.
  ```

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha) = prod_j x_j**(alpha_j - 1) / Z
  Z = prod_j Gamma(alpha_j) / Gamma(sum_j alpha_j)
  ```

  where:

  * `x in S^{k-1}`, i.e., the `(k-1)`-simplex,
  * `concentration = alpha = [alpha_0, ..., alpha_{k-1}]`, `alpha_j > 0`,
  * `Z` is the normalization constant aka the [multivariate beta function](
    https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function),
    and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The `concentration` represents mean total counts of class occurrence, i.e.,

  ```none
  concentration = alpha = mean * total_concentration
  ```

  where `mean` in `S^{k-1}` and `total_concentration` is a positive real number
  representing a mean total count.

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  Warning: Some components of the samples can be zero due to finite precision.
  This happens more often when some of the concentrations are very small.
  Make sure to round the samples to `np.finfo(dtype).tiny` before computing the
  density.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in
  (Figurnov et al., 2018).

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Create a single trivariate Dirichlet, with the 3rd class being three times
  # more frequent than the first. I.e., batch_shape=[], event_shape=[3].
  alpha = [1., 2, 3]
  dist = tfd.Dirichlet(alpha)

  dist.sample([4, 5])  # shape: [4, 5, 3]

  # x has one sample, one batch, three classes:
  x = [.2, .3, .5]   # shape: [3]
  dist.prob(x)       # shape: []

  # x has two samples from one batch:
  x = [[.1, .4, .5],
       [.2, .3, .5]]
  dist.prob(x)         # shape: [2]

  # alpha will be broadcast to shape [5, 7, 3] to match x.
  x = [[...]]   # shape: [5, 7, 3]
  dist.prob(x)  # shape: [5, 7]
  ```

  ```python
  # Create batch_shape=[2], event_shape=[3]:
  alpha = [[1., 2, 3],
           [4, 5, 6]]   # shape: [2, 3]
  dist = tfd.Dirichlet(alpha)

  dist.sample([4, 5])  # shape: [4, 5, 2, 3]

  x = [.2, .3, .5]
  # x will be broadcast as [[.2, .3, .5],
  #                         [.2, .3, .5]],
  # thus matching batch_shape [2, 3].
  dist.prob(x)         # shape: [2]
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  alpha = tf.constant([1.0, 2.0, 3.0])
  dist = tfd.Dirichlet(alpha)
  samples = dist.sample(5)  # Shape [5, 3]
  loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tf.gradients(loss, alpha)
  ```

  References:
    Implicit Reparameterization Gradients:
      [Figurnov et al., 2018]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
      ([pdf]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
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
               validate_args=False,
               allow_nan_stats=True,
               name="Dirichlet"):
    # [FUNCTION DOC]
    # Constructor. Converts concentration to a tensor (optionally validated), stores it and its
    # sum over the last axis, then initializes the base Distribution with the concentration
    # dtype, fully reparameterized sampling, and the stored parameters.
    # Inputs: concentration is a positive floating-point array-like with rank at least 1 (last
    # dimension is the event size, must be at least 2 when validating); validate_args (default
    # false) adds runtime assertions of positivity, rank and event size; allow_nan_stats
    # (default true) makes undefined statistics return NaN instead of raising; name is the
    # name scope prefix (default "Dirichlet").
    # Outputs: none. Side effects: creates ops in a name scope; the deprecation decorator logs
    # a one-time warning. Raises a runtime assertion error at execution (or graph build time
    # for static values) only when validate_args is true and the concentration is invalid.
    """Initialize a batch of Dirichlet distributions.

    Args:
      concentration: Positive floating-point `Tensor` indicating mean number
        of class occurrences; aka "alpha". Implies `self.dtype`, and
        `self.batch_shape`, `self.event_shape`, i.e., if
        `concentration.shape = [N1, N2, ..., Nm, k]` then
        `batch_shape = [N1, N2, ..., Nm]` and
        `event_shape = [k]`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with ops.name_scope(name, values=[concentration]) as name:
      self._concentration = self._maybe_assert_valid_concentration(
          ops.convert_to_tensor(concentration, name="concentration"),
          validate_args)
      self._total_concentration = math_ops.reduce_sum(self._concentration, -1)
    super(Dirichlet, self).__init__(
        dtype=self._concentration.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=distribution.FULLY_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[self._concentration,
                       self._total_concentration],
        name=name)

  @property
  def concentration(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored concentration tensor (the alpha parameters, last
    # axis is the event dimension). No side effects.
    """Concentration parameter; expected counts for that coordinate."""
    return self._concentration

  @property
  def total_concentration(self):
    # [FUNCTION DOC]
    # Read-only property returning the tensor of concentration sums over the last axis,
    # computed once in the constructor. Its shape equals the batch shape. No side effects.
    """Sum of last dim of concentration parameter."""
    return self._total_concentration

  def _batch_shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public batch shape tensor method.
    # Returns the dynamic shape (integer vector tensor) of the total concentration, i.e. the
    # concentration shape without the event dimension. No inputs, no side effects.
    return array_ops.shape(self.total_concentration)

  def _batch_shape(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public static batch shape property.
    # Returns the static shape object of the total concentration (possibly partially unknown).
    return self.total_concentration.get_shape()

  def _event_shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public event shape tensor method.
    # Returns a one-element integer vector tensor holding the size of the last concentration
    # dimension (the number of classes).
    return array_ops.shape(self.concentration)[-1:]

  def _event_shape(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public static event shape property.
    # Returns the static shape of the last concentration dimension as a one-dimensional shape.
    # Raises a value error if the concentration has unknown rank or rank below 1.
    return self.concentration.get_shape().with_rank_at_least(1)[-1:]

  def _sample_n(self, n, seed=None):
    # [FUNCTION DOC]
    # HELPER: Base-class hook used by the public sample method.
    # Draws n samples by sampling independent gamma variates with the concentration as shape
    # (unit rate) and normalizing them by their sum over the last axis. Inputs: n is the
    # number of samples (scalar); seed is an optional random seed. Returns a tensor of shape
    # [n] + batch shape + event shape with the distribution dtype. Uses the random generator.
    gamma_sample = random_ops.random_gamma(
        shape=[n],
        alpha=self.concentration,
        dtype=self.dtype,
        seed=seed)
    return gamma_sample / math_ops.reduce_sum(gamma_sample, -1, keepdims=True)

  @distribution_util.AppendDocstring(_dirichlet_sample_note)
  def _log_prob(self, x):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public log probability method (its docstring gets an
    # appended note about valid values).
    # Returns the unnormalized log density minus the log normalization constant for the value
    # tensor x (non-negative, summing to one over the last axis, broadcastable with the batch
    # and event shape). With validate_args set, invalid values trigger assertion errors.
    return self._log_unnormalized_prob(x) - self._log_normalization()

  @distribution_util.AppendDocstring(_dirichlet_sample_note)
  def _prob(self, x):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public probability method (docstring gets the same
    # appended note about valid values).
    # Returns the exponential of the log probability of the value tensor x; same input
    # requirements and assertions as the log probability hook.
    return math_ops.exp(self._log_prob(x))

  def _log_unnormalized_prob(self, x):
    # [FUNCTION DOC]
    # HELPER: Part of the log density computation.
    # Optionally validates the sample x (positive and summing to one when validate_args is
    # set), then returns the sum over the last axis of (concentration - 1) times log x, using
    # a safe product that yields zero where the multiplier is zero. Returns a tensor with the
    # batch shape (after broadcasting).
    x = self._maybe_assert_valid_sample(x)
    return math_ops.reduce_sum(math_ops.xlogy(self.concentration - 1., x), -1)

  def _log_normalization(self):
    # [FUNCTION DOC]
    # HELPER: Part of the log density and entropy computation.
    # Returns the log of the multivariate beta function of the concentration (log of the
    # normalization constant Z), reduced over the last axis, so it has the batch shape.
    return special_math_ops.lbeta(self.concentration)

  def _entropy(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public entropy method.
    # Computes the closed-form differential entropy: log normalization plus
    # (total concentration minus event size) times digamma of total concentration, minus the sum
    # over classes of (concentration - 1) times digamma of concentration. Returns a tensor of
    # the batch shape with the distribution dtype.
    k = math_ops.cast(self.event_shape_tensor()[0], self.dtype)
    return (
        self._log_normalization()
        + ((self.total_concentration - k)
           * math_ops.digamma(self.total_concentration))
        - math_ops.reduce_sum(
            (self.concentration - 1.) * math_ops.digamma(self.concentration),
            axis=-1))

  def _mean(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public mean method.
    # Returns concentration divided by total concentration (broadcast over the last axis); the
    # result has batch shape plus event shape.
    return self.concentration / self.total_concentration[..., array_ops.newaxis]

  def _covariance(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public covariance method.
    # Builds the covariance matrix as the negative outer product of (scale term times mean)
    # with itself, then overwrites its diagonal with the variance. Returns a tensor of shape
    # batch shape + [k, k], where k is the event size.
    x = self._variance_scale_term() * self._mean()
    # pylint: disable=invalid-unary-operand-type
    return array_ops.matrix_set_diag(
        -math_ops.matmul(
            x[..., array_ops.newaxis],
            x[..., array_ops.newaxis, :]),  # outer prod
        self._variance())

  def _variance(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public variance method.
    # Computes x = scale * mean, where scale is the shared scale term, and returns
    # x * (scale - x) per class. Output has batch shape plus event shape.
    scale = self._variance_scale_term()
    x = scale * self._mean()
    return x * (scale - x)

  def _variance_scale_term(self):
    # [FUNCTION DOC]
    # HELPER: Shared by the covariance and variance computations.
    # Returns the reciprocal square root of (1 + total concentration), with a trailing axis
    # added so it broadcasts against the event dimension.
    """Helper to `_covariance` and `_variance` which computes a shared scale."""
    return math_ops.rsqrt(1. + self.total_concentration[..., array_ops.newaxis])

  @distribution_util.AppendDocstring(
      """Note: The mode is undefined when any `concentration <= 1`. If
      `self.allow_nan_stats` is `True`, `NaN` is used for undefined modes. If
      `self.allow_nan_stats` is `False` an exception is raised when one or more
      modes are undefined.""")
  def _mode(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public mode method (docstring gets an appended note).
    # Computes (concentration - 1) / (total concentration - event size). The mode is defined
    # only when every concentration in the event exceeds 1. If allow_nan_stats is true, batch
    # members where any concentration is not above 1 get NaN; otherwise the result is
    # wrapped with a dependency on an assertion that raises an invalid-argument error when
    # the condition fails (assertion is on the concentration elementwise). Returns a tensor of
    # batch shape plus event shape.
    k = math_ops.cast(self.event_shape_tensor()[0], self.dtype)
    mode = (self.concentration - 1.) / (
        self.total_concentration[..., array_ops.newaxis] - k)
    if self.allow_nan_stats:
      nan = array_ops.fill(
          array_ops.shape(mode),
          np.array(np.nan, dtype=self.dtype.as_numpy_dtype()),
          name="nan")
      return array_ops.where_v2(
          math_ops.reduce_all(self.concentration > 1., axis=-1), mode, nan)
    return control_flow_ops.with_dependencies([
        check_ops.assert_less(
            array_ops.ones([], self.dtype),
            self.concentration,
            message="Mode undefined when any concentration <= 1"),
    ], mode)

  def _maybe_assert_valid_concentration(self, concentration, validate_args):
    # [FUNCTION DOC]
    # HELPER: Constructor validation helper.
    # Inputs: concentration tensor; validate_args flag. If the flag is false, returns the
    # concentration unchanged. Otherwise returns it wrapped with control dependencies on
    # assertions that all entries are positive, rank is at least 1, and the last dimension is
    # at least 2; failing assertions raise invalid-argument errors when evaluated (or at
    # graph construction time for statically known values).
    """Checks the validity of the concentration parameter."""
    if not validate_args:
      return concentration
    return control_flow_ops.with_dependencies([
        check_ops.assert_positive(
            concentration,
            message="Concentration parameter must be positive."),
        check_ops.assert_rank_at_least(
            concentration, 1,
            message="Concentration parameter must have >=1 dimensions."),
        check_ops.assert_less(
            1, array_ops.shape(concentration)[-1],
            message="Concentration parameter must have event_size >= 2."),
    ], concentration)

  def _maybe_assert_valid_sample(self, x):
    # [FUNCTION DOC]
    # HELPER: Sample validation helper for the density methods.
    # Returns x unchanged when validate_args is false; otherwise returns x tied to assertions
    # that all entries are positive and that the last-axis sum is approximately one. Failed
    # assertions raise invalid-argument errors.
    """Checks the validity of a sample."""
    if not self.validate_args:
      return x
    return control_flow_ops.with_dependencies([
        check_ops.assert_positive(x, message="samples must be positive"),
        check_ops.assert_near(
            array_ops.ones([], dtype=self.dtype),
            math_ops.reduce_sum(x, -1),
            message="sample last-dimension must sum to `1`"),
    ], x)


@kullback_leibler.RegisterKL(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(d1, d2, name=None):
  # [FUNCTION DOC]
  # HELPER: Registered with the KL registry for the pair of Dirichlet types; used by the
  # generic KL divergence function, not called directly.
  # Computes the closed-form KL divergence from d1 to d2: the sum over classes of
  # (a1 - a2) * (digamma(a1) - digamma(sum a1)) minus lbeta(a1) plus lbeta(a2), where a1, a2
  # are the concentrations. Inputs: d1 and d2 Dirichlet instances with broadcast-compatible
  # batch shapes; name optional (default "kl_dirichlet_dirichlet"). Returns a tensor of the
  # batch shape. Creates ops in a name scope.
  """Batchwise KL divergence KL(d1 || d2) with d1 and d2 Dirichlet.

  Args:
    d1: instance of a Dirichlet distribution object.
    d2: instance of a Dirichlet distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_dirichlet_dirichlet".

  Returns:
    Batchwise KL(d1 || d2)
  """
  with ops.name_scope(name, "kl_dirichlet_dirichlet", values=[
      d1.concentration, d2.concentration]):
    # The KL between Dirichlet distributions can be derived as follows. We have
    #
    #   Dir(x; a) = 1 / B(a) * prod_i[x[i]^(a[i] - 1)]
    #
    # where B(a) is the multivariate Beta function:
    #
    #   B(a) = Gamma(a[1]) * ... * Gamma(a[n]) / Gamma(a[1] + ... + a[n])
    #
    # The KL is
    #
    #   KL(Dir(x; a), Dir(x; b)) = E_Dir(x; a){log(Dir(x; a) / Dir(x; b))}
    #
    # so we'll need to know the log density of the Dirichlet. This is
    #
    #   log(Dir(x; a)) = sum_i[(a[i] - 1) log(x[i])] - log B(a)
    #
    # The only term that matters for the expectations is the log(x[i]). To
    # compute the expectation of this term over the Dirichlet density, we can
    # use the following facts about the Dirichlet in exponential family form:
    #   1. log(x[i]) is a sufficient statistic
    #   2. expected sufficient statistics (of any exp family distribution) are
    #      equal to derivatives of the log normalizer with respect to
    #      corresponding natural parameters: E{T[i](x)} = dA/d(eta[i])
    #
    # To proceed, we can rewrite the Dirichlet density in exponential family
    # form as follows:
    #
    #   Dir(x; a) = exp{eta(a) . T(x) - A(a)}
    #
    # where '.' is the dot product of vectors eta and T, and A is a scalar:
    #
    #   eta[i](a) = a[i] - 1
    #     T[i](x) = log(x[i])
    #        A(a) = log B(a)
    #
    # Now, we can use fact (2) above to write
    #
    #   E_Dir(x; a)[log(x[i])]
    #       = dA(a) / da[i]
    #       = d/da[i] log B(a)
    #       = d/da[i] (sum_j lgamma(a[j])) - lgamma(sum_j a[j])
    #       = digamma(a[i])) - digamma(sum_j a[j])
    #
    # Putting it all together, we have
    #
    # KL[Dir(x; a) || Dir(x; b)]
    #     = E_Dir(x; a){log(Dir(x; a) / Dir(x; b)}
    #     = E_Dir(x; a){sum_i[(a[i] - b[i]) log(x[i])} - (lbeta(a) - lbeta(b))
    #     = sum_i[(a[i] - b[i]) * E_Dir(x; a){log(x[i])}] - lbeta(a) + lbeta(b)
    #     = sum_i[(a[i] - b[i]) * (digamma(a[i]) - digamma(sum_j a[j]))]
    #          - lbeta(a) + lbeta(b))

    digamma_sum_d1 = math_ops.digamma(
        math_ops.reduce_sum(d1.concentration, axis=-1, keepdims=True))
    digamma_diff = math_ops.digamma(d1.concentration) - digamma_sum_d1
    concentration_diff = d1.concentration - d2.concentration

    return (math_ops.reduce_sum(concentration_diff * digamma_diff, axis=-1) -
            special_math_ops.lbeta(d1.concentration) +
            special_math_ops.lbeta(d2.concentration))
