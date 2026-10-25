# [MODULE DOC]
# Defines the Multinomial distribution of the legacy tf.distributions API (v1 only): a
# distribution over vectors of non-negative class counts summing to a total count, parameterized
# by logits or probabilities. Also holds a module-level documentation note string appended to the
# log-probability docs. Samples are generated per batch member with the categorical sampling op
# and aggregated into counts. Import-time effect: exports Multinomial under the v1 distributions
# namespace.
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
"""The Multinomial distribution class."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


__all__ = [
    "Multinomial",
]


_multinomial_sample_note = """For each batch of counts, `value = [n_0, ...
,n_{k-1}]`, `P[value]` is the probability that after sampling `self.total_count`
draws from this Multinomial distribution, the number of draws falling in class
`j` is `n_j`. Since this definition is [exchangeable](
https://en.wikipedia.org/wiki/Exchangeable_random_variables); different
sequences have the same counts so the probability includes a combinatorial
coefficient.

Note: `value` must be a non-negative tensor with dtype `self.dtype`, have no
fractional components, and such that
`tf.reduce_sum(value, -1) = self.total_count`. Its shape must be broadcastable
with `self.probs` and `self.total_count`."""


@tf_export(v1=["distributions.Multinomial"])
class Multinomial(distribution.Distribution):
  # [CLASS DOC]
  # Multinomial distribution over length-K count vectors with a given number of trials per draw.
  # Holds total_count, logits, and probs tensors (logits and probs are derived from each other)
  # plus a cached mean tensor (total_count times probs) used to derive batch and event shapes.
  # Batch shape is everything but the last axis, event shape is the class axis. Samples are not
  # reparameterized. Supports log probability, mean, variance and covariance. Deprecated.
  """Multinomial distribution.

  This Multinomial distribution is parameterized by `probs`, a (batch of)
  length-`K` `prob` (probability) vectors (`K > 1`) such that
  `tf.reduce_sum(probs, -1) = 1`, and a `total_count` number of trials, i.e.,
  the number of trials per draw from the Multinomial. It is defined over a
  (batch of) length-`K` vector `counts` such that
  `tf.reduce_sum(counts, -1) = total_count`. The Multinomial is identically the
  Binomial distribution when `K = 2`.

  #### Mathematical Details

  The Multinomial is a distribution over `K`-class counts, i.e., a length-`K`
  vector of non-negative integer `counts = n = [n_0, ..., n_{K-1}]`.

  The probability mass function (pmf) is,

  ```none
  pmf(n; pi, N) = prod_j (pi_j)**n_j / Z
  Z = (prod_j n_j!) / N!
  ```

  where:
  * `probs = pi = [pi_0, ..., pi_{K-1}]`, `pi_j > 0`, `sum_j pi_j = 1`,
  * `total_count = N`, `N` a positive integer,
  * `Z` is the normalization constant, and,
  * `N!` denotes `N` factorial.

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

  Create a 3-class distribution, with the 3rd class is most likely to be drawn,
  using logits.

  ```python
  logits = [-50., -43, 0]
  dist = Multinomial(total_count=4., logits=logits)
  ```

  Create a 3-class distribution, with the 3rd class is most likely to be drawn.

  ```python
  p = [.2, .3, .5]
  dist = Multinomial(total_count=4., probs=p)
  ```

  The distribution functions can be evaluated on counts.

  ```python
  # counts same shape as p.
  counts = [1., 0, 3]
  dist.prob(counts)  # Shape []

  # p will be broadcast to [[.2, .3, .5], [.2, .3, .5]] to match counts.
  counts = [[1., 2, 1], [2, 2, 0]]
  dist.prob(counts)  # Shape [2]

  # p will be broadcast to shape [5, 7, 3] to match counts.
  counts = [[...]]  # Shape [5, 7, 3]
  dist.prob(counts)  # Shape [5, 7]
  ```

  Create a 2-batch of 3-class distributions.

  ```python
  p = [[.1, .2, .7], [.3, .3, .4]]  # Shape [2, 3]
  dist = Multinomial(total_count=[4., 5], probs=p)

  counts = [[2., 1, 1], [3, 1, 1]]
  dist.prob(counts)  # Shape [2]

  dist.sample(5) # Shape [5, 2, 3]
  ```
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
               total_count,
               logits=None,
               probs=None,
               validate_args=False,
               allow_nan_stats=True,
               name="Multinomial"):
    # [FUNCTION DOC]
    # Constructor. total_count: non-negative floating tensor of trial counts (integer valued);
    # logits or probs: exactly one should be given, floating tensors whose last axis indexes the K
    # classes and which share the dtype of total_count; validate_args (bool, default false) adds
    # checks that total_count is a non-negative integer form and that the probabilities are valid;
    # allow_nan_stats (bool, default true); name (scope name). Converts total_count to a tensor,
    # derives both logits and probs via the shared helper (which raises a value error if neither
    # or both are supplied), and caches the mean as total_count times probs. Initializes the base
    # class with non-reparameterized sampling. Deprecated; warns once.
    """Initialize a batch of Multinomial distributions.

    Args:
      total_count: Non-negative floating point tensor with shape broadcastable
        to `[N1,..., Nm]` with `m >= 0`. Defines this as a batch of
        `N1 x ... x Nm` different Multinomial distributions. Its components
        should be equal to integer values.
      logits: Floating point tensor representing unnormalized log-probabilities
        of a positive event with shape broadcastable to
        `[N1,..., Nm, K]` `m >= 0`, and the same dtype as `total_count`. Defines
        this as a batch of `N1 x ... x Nm` different `K` class Multinomial
        distributions. Only one of `logits` or `probs` should be passed in.
      probs: Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm, K]` `m >= 0` and same dtype as `total_count`. Defines
        this as a batch of `N1 x ... x Nm` different `K` class Multinomial
        distributions. `probs`'s components in the last portion of its shape
        should sum to `1`. Only one of `logits` or `probs` should be passed in.
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
    with ops.name_scope(name, values=[total_count, logits, probs]) as name:
      self._total_count = ops.convert_to_tensor(total_count, name="total_count")
      if validate_args:
        self._total_count = (
            distribution_util.embed_check_nonnegative_integer_form(
                self._total_count))
      self._logits, self._probs = distribution_util.get_logits_and_probs(
          logits=logits,
          probs=probs,
          multidimensional=True,
          validate_args=validate_args,
          name=name)
      self._mean_val = self._total_count[..., array_ops.newaxis] * self._probs
    super(Multinomial, self).__init__(
        dtype=self._probs.dtype,
        reparameterization_type=distribution.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._total_count,
                       self._logits,
                       self._probs],
        name=name)

  @property
  def total_count(self):
    # [FUNCTION DOC]
    # Property returning the tensor of per-draw trial counts (validated if requested).
    """Number of trials used to construct a sample."""
    return self._total_count

  @property
  def logits(self):
    # [FUNCTION DOC]
    # Property returning the unnormalized log-probability tensor (last axis is the classes).
    """Vector of coordinatewise logits."""
    return self._logits

  @property
  def probs(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says "probability of drawing a 1 in that coordinate", which is
    # copied from a binary distribution; this is actually the per-class probability vector.
    # Property returning the class probability tensor (last axis is the classes).
    """Probability of drawing a `1` in that coordinate."""
    return self._probs

  def _batch_shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class batch_shape_tensor.
    # Returns the dynamic shape of the cached mean tensor without its last (class) axis.
    return array_ops.shape(self._mean_val)[:-1]

  def _batch_shape(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class batch_shape.
    # Returns the static shape of the cached mean tensor without its last axis; requires the mean
    # to have rank of at least one.
    return self._mean_val.get_shape().with_rank_at_least(1)[:-1]

  def _event_shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class event_shape_tensor.
    # Returns the dynamic length of the last axis of the cached mean tensor, as a one-element
    # shape (the number of classes K).
    return array_ops.shape(self._mean_val)[-1:]

  def _event_shape(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class event_shape.
    # Returns the static last-axis shape of the cached mean tensor (the number of classes K);
    # requires rank of at least one.
    return self._mean_val.get_shape().with_rank_at_least(1)[-1:]

  def _sample_n(self, n, seed=None):
    # [FUNCTION DOC]
    # HELPER: hook for the base class sample.
    # Draws n count vectors per batch member. n is an integer count of samples; seed is an
    # optional integer random seed (default none). Casts total_count to 32-bit integers,
    # broadcasts total_count and logits to a common batch shape, flattens the batch, and for each
    # flattened batch member draws n times the trial count categorical samples with the
    # multinomial sampling op (mapped over the batch), turns them into one-hot vectors and sums
    # them into per-class counts. Finally transposes and reshapes to n followed by the batch
    # shape and K. Returns a tensor of the distribution dtype. Consumes randomness.
    n_draws = math_ops.cast(self.total_count, dtype=dtypes.int32)
    k = self.event_shape_tensor()[0]

    # broadcast the total_count and logits to same shape
    n_draws = array_ops.ones_like(
        self.logits[..., 0], dtype=n_draws.dtype) * n_draws
    logits = array_ops.ones_like(
        n_draws[..., array_ops.newaxis], dtype=self.logits.dtype) * self.logits

    # flatten the total_count and logits
    flat_logits = array_ops.reshape(logits, [-1, k])  # [B1B2...Bm, k]
    flat_ndraws = n * array_ops.reshape(n_draws, [-1])  # [B1B2...Bm]

    # computes each total_count and logits situation by map_fn
    def _sample_single(args):
      # [FUNCTION DOC]
      # HELPER: per-batch-member sampler passed to the map function inside the sampling hook.
      # args is a pair of a logits vector of length K and a scalar number of draws (already
      # multiplied by the sample count n). Draws that many categorical samples from the logits,
      # reshapes them to n rows, converts to one-hot vectors and sums over the draw axis.
      # Returns a tensor of counts shaped n by K.
      logits, n_draw = args[0], args[1]  # [K], []
      x = random_ops.multinomial(logits[array_ops.newaxis, ...], n_draw,
                                 seed)  # [1, n*n_draw]
      x = array_ops.reshape(x, shape=[n, -1])  # [n, n_draw]
      x = math_ops.reduce_sum(array_ops.one_hot(x, depth=k), axis=-2)  # [n, k]
      return x

    x = map_fn.map_fn(
        _sample_single, [flat_logits, flat_ndraws],
        dtype=self.dtype)  # [B1B2...Bm, n, k]

    # reshape the results to proper shape
    x = array_ops.transpose(x, perm=[1, 0, 2])
    final_shape = array_ops.concat([[n], self.batch_shape_tensor(), [k]], 0)
    x = array_ops.reshape(x, final_shape)  # [n, B1, B2,..., Bm, k]
    return x

  @distribution_util.AppendDocstring(_multinomial_sample_note)
  def _log_prob(self, counts):
    # [FUNCTION DOC]
    # HELPER: hook for the base class log_prob.
    # counts is a tensor of non-negative integer-valued counts whose last axis sums to
    # total_count. Returns the log probability mass: unnormalized log probability minus the log
    # normalization. The appended note documents the expected shape and constraints. When
    # validate_args is set, invalid counts trigger runtime assertion failures.
    return self._log_unnormalized_prob(counts) - self._log_normalization(counts)

  def _log_unnormalized_prob(self, counts):
    # [FUNCTION DOC]
    # HELPER: part of the log mass computation.
    # Validates counts, then returns the sum over classes of counts times log-softmax of logits.
    counts = self._maybe_assert_valid_sample(counts)
    return math_ops.reduce_sum(counts * nn_ops.log_softmax(self.logits), -1)

  def _log_normalization(self, counts):
    # [FUNCTION DOC]
    # HELPER: part of the log mass computation.
    # Validates counts, then returns the negative log of the multinomial coefficient (log number
    # of arrangements) computed from total_count and counts.
    counts = self._maybe_assert_valid_sample(counts)
    return -distribution_util.log_combinations(self.total_count, counts)

  def _mean(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class mean.
    # Returns an identity copy of the cached mean (total_count times probs), shape batch by K.
    return array_ops.identity(self._mean_val)

  def _covariance(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class covariance.
    # Builds the K by K covariance matrix per batch member: minus the outer product of the mean
    # and the (broadcast) probabilities, with the diagonal replaced by the variance. Returns a
    # tensor shaped batch by K by K.
    p = self.probs * array_ops.ones_like(
        self.total_count)[..., array_ops.newaxis]
    # pylint: disable=invalid-unary-operand-type
    return array_ops.matrix_set_diag(
        -math_ops.matmul(
            self._mean_val[..., array_ops.newaxis],
            p[..., array_ops.newaxis, :]),  # outer product
        self._variance())

  def _variance(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class variance.
    # Returns per-class variance: mean minus mean times probability (the binomial variance
    # formula), with probabilities broadcast against total_count. Shape batch by K.
    p = self.probs * array_ops.ones_like(
        self.total_count)[..., array_ops.newaxis]
    return self._mean_val - self._mean_val * p

  def _maybe_assert_valid_sample(self, counts):
    # [FUNCTION DOC]
    # HELPER: input validation for the log mass computations.
    # KNOWN ISSUE: the docstring says it returns a tensor version of counts, but when
    # validate_args is false it returns counts untouched (no conversion or checking at all).
    # With validate_args true, asserts counts is non-negative integer valued and that it sums
    # along the last axis to total_count (runtime assertion failure otherwise), and returns counts
    # made dependent on those checks.
    """Check counts for proper shape, values, then return tensor version."""
    if not self.validate_args:
      return counts
    counts = distribution_util.embed_check_nonnegative_integer_form(counts)
    return control_flow_ops.with_dependencies([
        check_ops.assert_equal(
            self.total_count, math_ops.reduce_sum(counts, -1),
            message="counts must sum to `self.total_count`"),
    ], counts)
