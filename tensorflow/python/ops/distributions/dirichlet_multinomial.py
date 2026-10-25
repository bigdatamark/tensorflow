# [MODULE DOC]
# Defines the deprecated (v1 only) Dirichlet-Multinomial compound distribution over vectors of
# non-negative integer class counts that sum to a given total count, with class probabilities
# drawn from a Dirichlet. Exposes the class DirichletMultinomial through the module export list
# and builds on the generic Distribution base class and the distribution utility helpers.
# Import-time side effects: registers the class with the public API export decorator.
# Also defines a shared documentation note appended to the probability-method docstrings.
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
"""The DirichletMultinomial distribution class."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


__all__ = [
    "DirichletMultinomial",
]


_dirichlet_multinomial_sample_note = """For each batch of counts,
`value = [n_0, ..., n_{K-1}]`, `P[value]` is the probability that after
sampling `self.total_count` draws from this Dirichlet-Multinomial distribution,
the number of draws falling in class `j` is `n_j`. Since this definition is
[exchangeable](https://en.wikipedia.org/wiki/Exchangeable_random_variables);
different sequences have the same counts so the probability includes a
combinatorial coefficient.

Note: `value` must be a non-negative tensor with dtype `self.dtype`, have no
fractional components, and such that
`tf.reduce_sum(value, -1) = self.total_count`. Its shape must be broadcastable
with `self.concentration` and `self.total_count`."""


@tf_export(v1=["distributions.DirichletMultinomial"])
class DirichletMultinomial(distribution.Distribution):
  # [CLASS DOC]
  # Dirichlet-Multinomial compound distribution (a Beta-Binomial generalization) over count
  # vectors whose last dimension is the number of classes K (at least 2) and whose entries sum
  # to total_count. Holds three tensors: total_count (number of trials per draw), concentration
  # (positive Dirichlet parameters, last axis is the event dimension) and total_concentration
  # (concentration summed over the last axis). Subclasses the generic Distribution, with samples
  # that are not reparameterized, and implements the private hooks (shapes, sampling, log/prob,
  # mean, covariance, variance) used by the public methods. Created from array-likes for total
  # count and concentration; deprecated in favor of the separate probability library.
  """Dirichlet-Multinomial compound distribution.

  The Dirichlet-Multinomial distribution is parameterized by a (batch of)
  length-`K` `concentration` vectors (`K > 1`) and a `total_count` number of
  trials, i.e., the number of trials per draw from the DirichletMultinomial. It
  is defined over a (batch of) length-`K` vector `counts` such that
  `tf.reduce_sum(counts, -1) = total_count`. The Dirichlet-Multinomial is
  identically the Beta-Binomial distribution when `K = 2`.

  #### Mathematical Details

  The Dirichlet-Multinomial is a distribution over `K`-class counts, i.e., a
  length-`K` vector of non-negative integer `counts = n = [n_0, ..., n_{K-1}]`.

  The probability mass function (pmf) is,

  ```none
  pmf(n; alpha, N) = Beta(alpha + n) / (prod_j n_j!) / Z
  Z = Beta(alpha) / N!
  ```

  where:

  * `concentration = alpha = [alpha_0, ..., alpha_{K-1}]`, `alpha_j > 0`,
  * `total_count = N`, `N` a positive integer,
  * `N!` is `N` factorial, and,
  * `Beta(x) = prod_j Gamma(x_j) / Gamma(sum_j x_j)` is the
    [multivariate beta function](
    https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function),
    and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  Dirichlet-Multinomial is a [compound distribution](
  https://en.wikipedia.org/wiki/Compound_probability_distribution), i.e., its
  samples are generated as follows.

    1. Choose class probabilities:
       `probs = [p_0,...,p_{K-1}] ~ Dir(concentration)`
    2. Draw integers:
       `counts = [n_0,...,n_{K-1}] ~ Multinomial(total_count, probs)`

  The last `concentration` dimension parametrizes a single Dirichlet-Multinomial
  distribution. When calling distribution functions (e.g., `dist.prob(counts)`),
  `concentration`, `total_count` and `counts` are broadcast to the same shape.
  The last dimension of `counts` corresponds single Dirichlet-Multinomial
  distributions.

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  #### Pitfalls

  The number of classes, `K`, must not exceed:
  - the largest integer representable by `self.dtype`, i.e.,
    `2**(mantissa_bits+1)` (IEE754),
  - the maximum `Tensor` index, i.e., `2**31-1`.

  In other words,

  ```python
  K <= min(2**31-1, {
    tf.float16: 2**11,
    tf.float32: 2**24,
    tf.float64: 2**53 }[param.dtype])
  ```

  Note: This condition is validated only when `self.validate_args = True`.

  #### Examples

  ```python
  alpha = [1., 2., 3.]
  n = 2.
  dist = DirichletMultinomial(n, alpha)
  ```

  Creates a 3-class distribution, with the 3rd class is most likely to be
  drawn.
  The distribution functions can be evaluated on counts.

  ```python
  # counts same shape as alpha.
  counts = [0., 0., 2.]
  dist.prob(counts)  # Shape []

  # alpha will be broadcast to [[1., 2., 3.], [1., 2., 3.]] to match counts.
  counts = [[1., 1., 0.], [1., 0., 1.]]
  dist.prob(counts)  # Shape [2]

  # alpha will be broadcast to shape [5, 7, 3] to match counts.
  counts = [[...]]  # Shape [5, 7, 3]
  dist.prob(counts)  # Shape [5, 7]
  ```

  Creates a 2-batch of 3-class distributions.

  ```python
  alpha = [[1., 2., 3.], [4., 5., 6.]]  # Shape [2, 3]
  n = [3., 3.]
  dist = DirichletMultinomial(n, alpha)

  # counts will be broadcast to [[2., 1., 0.], [2., 1., 0.]] to match alpha.
  counts = [2., 1., 0.]
  dist.prob(counts)  # Shape [2]
  ```

  """

  # TODO(b/27419586) Change docstring for dtype of concentration once int
  # allowed.
  @deprecation.deprecated(
      "2019-01-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.distributions`.",
      warn_once=True)
  def __init__(self,
               total_count,
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name="DirichletMultinomial"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the concentration dtype matches "n", but no parameter
    # named n exists (it means total_count); the docstring also implies total_count may be
    # broadcast freely, yet sampling only supports a scalar total count (see sampling hook).
    # Constructor. Converts total_count and concentration to tensors; when validate_args is
    # true, requires total_count to be non-negative integer valued and concentration to have a
    # valid categorical event shape and be positive. Stores total_count, concentration and the
    # concentration sum over the last axis, and initializes the base Distribution with the
    # concentration dtype and no reparameterization.
    # Inputs: total_count (non-negative float array-like, broadcastable to the batch shape);
    # concentration (positive float array-like, last dimension is the class count); validate_args
    # (default false); allow_nan_stats (default true); name (default "DirichletMultinomial").
    # Outputs: none. Side effects: ops created in a name scope; deprecation decorator logs a
    # one-time warning. Raises assertion/invalid-argument errors only when validate_args is true.
    """Initialize a batch of DirichletMultinomial distributions.

    Args:
      total_count:  Non-negative floating point tensor, whose dtype is the same
        as `concentration`. The shape is broadcastable to `[N1,..., Nm]` with
        `m >= 0`. Defines this as a batch of `N1 x ... x Nm` different
        Dirichlet multinomial distributions. Its components should be equal to
        integer values.
      concentration: Positive floating point tensor, whose dtype is the
        same as `n` with shape broadcastable to `[N1,..., Nm, K]` `m >= 0`.
        Defines this as a batch of `N1 x ... x Nm` different `K` class Dirichlet
        multinomial distributions.
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
    with ops.name_scope(name, values=[total_count, concentration]) as name:
      # Broadcasting works because:
      # * The broadcasting convention is to prepend dimensions of size [1], and
      #   we use the last dimension for the distribution, whereas
      #   the batch dimensions are the leading dimensions, which forces the
      #   distribution dimension to be defined explicitly (i.e. it cannot be
      #   created automatically by prepending). This forces enough explicitness.
      # * All calls involving `counts` eventually require a broadcast between
      #  `counts` and concentration.
      self._total_count = ops.convert_to_tensor(total_count, name="total_count")
      if validate_args:
        self._total_count = (
            distribution_util.embed_check_nonnegative_integer_form(
                self._total_count))
      self._concentration = self._maybe_assert_valid_concentration(
          ops.convert_to_tensor(concentration,
                                name="concentration"),
          validate_args)
      self._total_concentration = math_ops.reduce_sum(self._concentration, -1)
    super(DirichletMultinomial, self).__init__(
        dtype=self._concentration.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=distribution.NOT_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[self._total_count,
                       self._concentration],
        name=name)

  @property
  def total_count(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored tensor of trial counts per draw. No side effects.
    """Number of trials used to construct a sample."""
    return self._total_count

  @property
  def concentration(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored concentration tensor (last axis is the class
    # dimension). No side effects.
    """Concentration parameter; expected prior counts for that coordinate."""
    return self._concentration

  @property
  def total_concentration(self):
    # [FUNCTION DOC]
    # Read-only property returning the concentration summed over the last axis, computed once in
    # the constructor. No side effects.
    """Sum of last dim of concentration parameter."""
    return self._total_concentration

  def _batch_shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public batch shape tensor method.
    # Returns the dynamic shape of the total concentration (concentration without its last
    # axis) as an integer vector tensor. Note total_count does not contribute to the batch shape.
    return array_ops.shape(self.total_concentration)

  def _batch_shape(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the static batch shape property.
    # Returns the static shape of the total concentration (possibly partially unknown).
    return self.total_concentration.get_shape()

  def _event_shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public event shape tensor method.
    # Returns a one-element integer vector tensor with the size of the last concentration
    # dimension (the number of classes).
    return array_ops.shape(self.concentration)[-1:]

  def _event_shape(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the static event shape property.
    # Returns the static shape of the last concentration dimension as a one-dimensional shape.
    # Raises a value error if the concentration rank is unknown or below 1.
    # Event shape depends only on total_concentration, not "n".
    return self.concentration.get_shape().with_rank_at_least(1)[-1:]

  def _sample_n(self, n, seed=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the number of draws is taken from the whole total_count tensor and passed as
    # the sample count of the multinomial op, which requires a scalar; a batched (non-scalar)
    # total_count will fail or misbehave even though the constructor docs allow broadcasting.
    # HELPER: Base-class hook used by the public sample method.
    # Draws n samples in two stages: gamma variates with the concentration as shape are
    # turned into log values and used as unnormalized logits (equivalent to sampling Dirichlet
    # class probabilities), then a multinomial op draws total_count categorical outcomes per
    # row, which are one-hot encoded and summed into class counts. Inputs: n is the number of
    # samples; seed is an optional random seed (a derived seed is used for the second stage).
    # Returns a tensor of shape [n] + batch shape + [K] with the distribution dtype.
    n_draws = math_ops.cast(self.total_count, dtype=dtypes.int32)
    k = self.event_shape_tensor()[0]
    unnormalized_logits = array_ops.reshape(
        math_ops.log(random_ops.random_gamma(
            shape=[n],
            alpha=self.concentration,
            dtype=self.dtype,
            seed=seed)),
        shape=[-1, k])
    draws = random_ops.multinomial(
        logits=unnormalized_logits,
        num_samples=n_draws,
        seed=distribution_util.gen_new_seed(seed, salt="dirichlet_multinomial"))
    x = math_ops.reduce_sum(array_ops.one_hot(draws, depth=k), -2)
    final_shape = array_ops.concat([[n], self.batch_shape_tensor(), [k]], 0)
    x = array_ops.reshape(x, final_shape)
    return math_ops.cast(x, self.dtype)

  @distribution_util.AppendDocstring(_dirichlet_multinomial_sample_note)
  def _log_prob(self, counts):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public log probability method (docstring gets an
    # appended note).
    # Optionally validates counts (non-negative integer form and last-axis sum equal to
    # total_count when validate_args is set), then returns lbeta(concentration + counts) minus
    # lbeta(concentration) plus the log of the combinatorial coefficient for total_count and
    # counts. Returns a tensor of the broadcast batch shape.
    counts = self._maybe_assert_valid_sample(counts)
    ordered_prob = (
        special_math_ops.lbeta(self.concentration + counts)
        - special_math_ops.lbeta(self.concentration))
    return ordered_prob + distribution_util.log_combinations(
        self.total_count, counts)

  @distribution_util.AppendDocstring(_dirichlet_multinomial_sample_note)
  def _prob(self, counts):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public probability method.
    # Returns the exponential of the log probability of the count tensor; same input
    # requirements and validation as the log probability hook.
    return math_ops.exp(self._log_prob(counts))

  def _mean(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public mean method.
    # Returns total_count times concentration divided by total concentration (broadcast on the
    # last axis); shape is the broadcast of batch shape plus class dimension.
    return self.total_count * (self.concentration /
                               self.total_concentration[..., array_ops.newaxis])

  @distribution_util.AppendDocstring(
      """The covariance for each batch member is defined as the following:

      ```none
      Var(X_j) = n * alpha_j / alpha_0 * (1 - alpha_j / alpha_0) *
      (n + alpha_0) / (1 + alpha_0)
      ```

      where `concentration = alpha` and
      `total_concentration = alpha_0 = sum_j alpha_j`.

      The covariance between elements in a batch is defined as:

      ```none
      Cov(X_i, X_j) = -n * alpha_i * alpha_j / alpha_0 ** 2 *
      (n + alpha_0) / (1 + alpha_0)
      ```
      """)
  def _covariance(self):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public covariance method (docstring gets an appended
    # formula note).
    # Builds the negative outer product of (scale term times mean) with itself and replaces its
    # diagonal with the variance. Returns a tensor of shape batch shape + [K, K].
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
    # With x = scale * mean, returns x * (total_count * scale - x) per class. Output shape is
    # batch shape plus class dimension.
    scale = self._variance_scale_term()
    x = scale * self._mean()
    return x * (self.total_count * scale - x)

  def _variance_scale_term(self):
    # [FUNCTION DOC]
    # HELPER: Shared by the covariance and variance computations.
    # Returns the square root of (1 + c0 / total_count) / (1 + c0), where c0 is the total
    # concentration with a trailing axis added for broadcasting against the class dimension.
    """Helper to `_covariance` and `_variance` which computes a shared scale."""
    # We must take care to expand back the last dim whenever we use the
    # total_concentration.
    c0 = self.total_concentration[..., array_ops.newaxis]
    return math_ops.sqrt((1. + c0 / self.total_count) / (1. + c0))

  def _maybe_assert_valid_concentration(self, concentration, validate_args):
    # [FUNCTION DOC]
    # HELPER: Constructor validation helper.
    # Inputs: concentration tensor; validate_args flag. If false, returns concentration as is.
    # Otherwise embeds a categorical event-shape check (rank at least 1, class count within the
    # dtype and index limits) and ties the result to an assertion that all entries are
    # positive; failures raise invalid-argument errors.
    """Checks the validity of the concentration parameter."""
    if not validate_args:
      return concentration
    concentration = distribution_util.embed_check_categorical_event_shape(
        concentration)
    return control_flow_ops.with_dependencies([
        check_ops.assert_positive(
            concentration,
            message="Concentration parameter must be positive."),
    ], concentration)

  def _maybe_assert_valid_sample(self, counts):
    # [FUNCTION DOC]
    # HELPER: Count validation helper for the density methods.
    # Returns counts unchanged if validate_args is false. Otherwise checks they are
    # non-negative integer valued and ties them to an assertion that the last-axis sum equals
    # total_count; failures raise invalid-argument errors. Returns the (possibly wrapped) counts.
    """Check counts for proper shape, values, then return tensor version."""
    if not self.validate_args:
      return counts
    counts = distribution_util.embed_check_nonnegative_integer_form(counts)
    return control_flow_ops.with_dependencies([
        check_ops.assert_equal(
            self.total_count, math_ops.reduce_sum(counts, -1),
            message="counts last-dimension must sum to `self.total_count`"),
    ], counts)
