# [MODULE DOC]
# Defines Student's t distribution of the legacy tf.distributions API (v1 only), a heavy-tailed
# location-scale distribution with degrees of freedom, location and scale, sampled via a normal
# variate divided by the root of a scaled gamma (chi-squared) variate with reparameterized
# gradients. Also defines StudentTWithAbsDfSoftplusScale, which floors the absolute value of df
# and applies softplus to scale. Import-time effect: exports StudentT under the v1 distributions
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
"""Student's t distribution class."""

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
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


__all__ = [
    "StudentT",
    "StudentTWithAbsDfSoftplusScale",
]


@tf_export(v1=["distributions.StudentT"])
class StudentT(distribution.Distribution):
  # [CLASS DOC]
  # Student's t distribution with degrees of freedom df, location loc and scale scale. Holds the
  # three parameters as identity-wrapped floating tensors that broadcast together; batch shape is
  # their broadcast shape, event shape is scalar, dtype comes from scale. Fully reparameterized
  # sampling. Supports density, cdf, entropy, mean, variance and mode, with mean and variance
  # undefined (NaN or an error) for df of at most one. Deprecated.
  """Student's t-distribution.

  This distribution has parameters: degree of freedom `df`, location `loc`,
  and `scale`.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; df, mu, sigma) = (1 + y**2 / df)**(-0.5 (df + 1)) / Z
  where,
  y = (x - mu) / sigma
  Z = abs(sigma) sqrt(df pi) Gamma(0.5 df) / Gamma(0.5 (df + 1))
  ```

  where:
  * `loc = mu`,
  * `scale = sigma`, and,
  * `Z` is the normalization constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The StudentT distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ StudentT(df, loc=0, scale=1)
  Y = loc + scale * X
  ```

  Notice that `scale` has semantics more similar to standard deviation than
  variance. However it is not actually the std. deviation; the Student's
  t-distribution std. dev. is `scale sqrt(df / (df - 2))` when `df > 2`.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in
  (Figurnov et al., 2018).

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Define a single scalar Student t distribution.
  single_dist = tfd.StudentT(df=3)

  # Evaluate the pdf at 1, returning a scalar Tensor.
  single_dist.prob(1.)

  # Define a batch of two scalar valued Student t's.
  # The first has degrees of freedom 2, mean 1, and scale 11.
  # The second 3, 2 and 22.
  multi_dist = tfd.StudentT(df=[2, 3], loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  multi_dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  multi_dist.sample(3)
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two Student's t distributions.
  # Both have df 2 and mean 1, but different scales.
  dist = tfd.StudentT(df=2, loc=1, scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  df = tf.constant(2.0)
  loc = tf.constant(2.0)
  scale = tf.constant(11.0)
  dist = tfd.StudentT(df=df, loc=loc, scale=scale)
  samples = dist.sample(5)  # Shape [5]
  loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tf.gradients(loss, [df, loc, scale])
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
               df,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="StudentT"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the type error is raised when loc and scale have different
    # dtypes, but the code checks all three of df, loc and scale for a common floating dtype. Also
    # validate_args only asserts that df is positive; scale positivity is never checked.
    # Constructor. df, loc, scale: floating tensors broadcastable together; validate_args (bool,
    # default false); allow_nan_stats (bool, default true); name (scope name). Stores identity
    # copies of the three parameters and initializes the base class as fully reparameterized with
    # the dtype of scale. Deprecated; warns once.
    """Construct Student's t distributions.

    The distributions have degree of freedom `df`, mean `loc`, and scale
    `scale`.

    The parameters `df`, `loc`, and `scale` must be shaped in a way that
    supports broadcasting (e.g. `df + loc + scale` is a valid operation).

    Args:
      df: Floating-point `Tensor`. The degrees of freedom of the
        distribution(s). `df` must contain only positive values.
      loc: Floating-point `Tensor`. The mean(s) of the distribution(s).
      scale: Floating-point `Tensor`. The scaling factor(s) for the
        distribution(s). Note that `scale` is not technically the standard
        deviation of this distribution but has semantics more similar to
        standard deviation than variance.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if loc and scale are different dtypes.
    """
    parameters = dict(locals())
    with ops.name_scope(name, values=[df, loc, scale]) as name:
      with ops.control_dependencies([check_ops.assert_positive(df)]
                                    if validate_args else []):
        self._df = array_ops.identity(df, name="df")
        self._loc = array_ops.identity(loc, name="loc")
        self._scale = array_ops.identity(scale, name="scale")
        check_ops.assert_same_float_dtype(
            (self._df, self._loc, self._scale))
    super(StudentT, self).__init__(
        dtype=self._scale.dtype,
        reparameterization_type=distribution.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._df, self._loc, self._scale],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    # [FUNCTION DOC]
    # HELPER: static helper for the base distribution's param_shapes machinery.
    # Returns a mapping from parameter names df, loc and scale to the sample shape converted to a
    # 32-bit integer tensor.
    return dict(
        zip(("df", "loc", "scale"), (
            [ops.convert_to_tensor(
                sample_shape, dtype=dtypes.int32)] * 3)))

  @property
  def df(self):
    # [FUNCTION DOC]
    # Property returning the degrees-of-freedom tensor.
    """Degrees of freedom in these Student's t distribution(s)."""
    return self._df

  @property
  def loc(self):
    # [FUNCTION DOC]
    # Property returning the location tensor.
    """Locations of these Student's t distribution(s)."""
    return self._loc

  @property
  def scale(self):
    # [FUNCTION DOC]
    # Property returning the scale tensor (not the standard deviation).
    """Scaling factors of these Student's t distribution(s)."""
    return self._scale

  def _batch_shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class batch_shape_tensor.
    # Returns the dynamic broadcast of the shapes of df, loc and scale as an integer tensor.
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.df),
        array_ops.broadcast_dynamic_shape(
            array_ops.shape(self.loc), array_ops.shape(self.scale)))

  def _batch_shape(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class batch_shape.
    # Returns the static broadcast of the shapes of df, loc and scale; raises a value error if
    # they are not broadcast compatible.
    return array_ops.broadcast_static_shape(
        array_ops.broadcast_static_shape(self.df.get_shape(),
                                         self.loc.get_shape()),
        self.scale.get_shape())

  def _event_shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class event_shape_tensor.
    # Returns an empty 32-bit integer vector since draws are scalars. (The dtype is looked up
    # through the math ops module rather than the dtypes module, which only works because that
    # module re-exposes it.)
    return constant_op.constant([], dtype=math_ops.int32)

  def _event_shape(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class event_shape.
    # Returns the static scalar (empty) event shape.
    return tensor_shape.TensorShape([])

  def _sample_n(self, n, seed=None):
    # [FUNCTION DOC]
    # HELPER: hook for the base class sample.
    # Draws n samples per batch member using the fact that a standard normal divided by the square
    # root of a chi-squared variate over its degrees of freedom is Student t. n is an integer
    # count; seed is an optional integer seed (default none). Draws normal noise, draws gamma
    # noise with shape half of df and rate one half (using a derived seed salted for this
    # distribution), forms the ratio, then scales by scale and shifts by loc (the absolute value
    # of scale is intentionally not used). Returns a tensor shaped n followed by the batch shape.
    # Consumes randomness.
    # The sampling method comes from the fact that if:
    #   X ~ Normal(0, 1)
    #   Z ~ Chi2(df)
    #   Y = X / sqrt(Z / df)
    # then:
    #   Y ~ StudentT(df).
    shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
    normal_sample = random_ops.random_normal(shape, dtype=self.dtype, seed=seed)
    df = self.df * array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype)
    gamma_sample = random_ops.random_gamma(
        [n],
        0.5 * df,
        beta=0.5,
        dtype=self.dtype,
        seed=distribution_util.gen_new_seed(seed, salt="student_t"))
    samples = normal_sample * math_ops.rsqrt(gamma_sample / df)
    return samples * self.scale + self.loc  # Abs(scale) not wanted.

  def _log_prob(self, x):
    # [FUNCTION DOC]
    # HELPER: hook for the base class log_prob.
    # Returns the unnormalized log density minus the log normalization for tensor x.
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _log_unnormalized_prob(self, x):
    # [FUNCTION DOC]
    # HELPER: part of the log density computation.
    # Standardizes x by loc and scale, then returns minus half of (df + 1) times log1p of the
    # squared standardized value divided by df.
    y = (x - self.loc) / self.scale  # Abs(scale) superfluous.
    return -0.5 * (self.df + 1.) * math_ops.log1p(y**2. / self.df)

  def _log_normalization(self):
    # [FUNCTION DOC]
    # HELPER: part of the log density computation.
    # Returns log of the absolute scale plus half log df plus half log pi plus log-gamma of half
    # df minus log-gamma of half of (df + 1): the log of the normalizing constant.
    return (math_ops.log(math_ops.abs(self.scale)) +
            0.5 * math_ops.log(self.df) +
            0.5 * np.log(np.pi) +
            math_ops.lgamma(0.5 * self.df) -
            math_ops.lgamma(0.5 * (self.df + 1.)))

  def _cdf(self, x):
    # [FUNCTION DOC]
    # HELPER: hook for the base class cdf.
    # Standardizes x using loc and the absolute value of scale, computes half of the regularized
    # incomplete beta function (parameters half df and one half) at df over (y squared plus df),
    # which is the lower tail, and returns that for negative standardized values and one minus it
    # otherwise. Returns a tensor of the batch shape.
    # Take Abs(scale) to make subsequent where work correctly.
    y = (x - self.loc) / math_ops.abs(self.scale)
    x_t = self.df / (y**2. + self.df)
    neg_cdf = 0.5 * math_ops.betainc(0.5 * self.df, 0.5, x_t)
    return array_ops.where_v2(math_ops.less(y, 0.), neg_cdf, 1. - neg_cdf)

  def _entropy(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class entropy.
    # Returns the differential entropy per batch member: log of absolute scale plus half log df
    # plus the log beta function of (df/2, 1/2) plus half of (df + 1) times the difference of
    # digamma at (df + 1)/2 and at df/2. Parameters are broadcast to the batch shape.
    v = array_ops.ones(self.batch_shape_tensor(),
                       dtype=self.dtype)[..., array_ops.newaxis]
    u = v * self.df[..., array_ops.newaxis]
    beta_arg = array_ops.concat([u, v], -1) / 2.
    return (math_ops.log(math_ops.abs(self.scale)) +
            0.5 * math_ops.log(self.df) +
            special_math_ops.lbeta(beta_arg) +
            0.5 * (self.df + 1.) *
            (math_ops.digamma(0.5 * (self.df + 1.)) -
             math_ops.digamma(0.5 * self.df)))

  @distribution_util.AppendDocstring(
      """The mean of Student's T equals `loc` if `df > 1`, otherwise it is
      `NaN`. If `self.allow_nan_stats=True`, then an exception will be raised
      rather than returning `NaN`.""")
  def _mean(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class mean.
    # KNOWN ISSUE: the appended docstring has the condition reversed: it says an exception is
    # raised when allow_nan_stats is true, but the code returns NaN in that case and only adds a
    # runtime assertion (failing when any df is at most 1) when allow_nan_stats is false.
    # Returns loc broadcast to the batch shape. With allow_nan_stats true, entries whose df is not
    # greater than one are replaced by NaN; otherwise the result depends on an assertion that
    # one is less than df.
    mean = self.loc * array_ops.ones(self.batch_shape_tensor(),
                                     dtype=self.dtype)
    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return array_ops.where_v2(
          math_ops.greater(
              self.df,
              array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype)),
          mean, array_ops.fill(self.batch_shape_tensor(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies(
          [
              check_ops.assert_less(
                  array_ops.ones([], dtype=self.dtype),
                  self.df,
                  message="mean not defined for components of df <= 1"),
          ],
          mean)

  @distribution_util.AppendDocstring("""
      The variance for Student's T equals

      ```
      df / (df - 2), when df > 2
      infinity, when 1 < df <= 2
      NaN, when df <= 1
      ```
      """)
  def _variance(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class variance.
    # Computes scale squared times df over (df - 2) where df exceeds two (using a safe
    # denominator so gradients never hit NaN), infinity where df is at most two, and then, if
    # allow_nan_stats is true, NaN where df is at most one. If allow_nan_stats is false, the
    # result depends on a runtime assertion that every df is greater than one. Returns a tensor
    # of the batch shape.
    # We need to put the tf.where inside the outer tf.where to ensure we never
    # hit a NaN in the gradient.
    denom = array_ops.where_v2(
        math_ops.greater(self.df, 2.), self.df - 2.,
        array_ops.ones_like(self.df))
    # Abs(scale) superfluous.
    var = (array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype) *
           math_ops.square(self.scale) * self.df / denom)
    # When 1 < df <= 2, variance is infinite.
    inf = np.array(np.inf, dtype=self.dtype.as_numpy_dtype())
    result_where_defined = array_ops.where_v2(
        self.df > array_ops.fill(self.batch_shape_tensor(), 2.), var,
        array_ops.fill(self.batch_shape_tensor(), inf, name="inf"))

    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return array_ops.where_v2(
          math_ops.greater(
              self.df,
              array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype)),
          result_where_defined,
          array_ops.fill(self.batch_shape_tensor(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies(
          [
              check_ops.assert_less(
                  array_ops.ones([], dtype=self.dtype),
                  self.df,
                  message="variance not defined for components of df <= 1"),
          ],
          result_where_defined)

  def _mode(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class mode.
    # Returns an identity copy of loc (not broadcast to the full batch shape).
    return array_ops.identity(self.loc)


class StudentTWithAbsDfSoftplusScale(StudentT):
  # [CLASS DOC]
  # Student's t distribution whose constructor maps df to floor of its absolute value and scale
  # through softplus so unconstrained reals can be supplied. Otherwise identical to StudentT.
  # Deprecated.
  """StudentT with `df = floor(abs(df))` and `scale = softplus(scale)`."""

  @deprecation.deprecated(
      "2019-01-01",
      "Use `tfd.StudentT(tf.floor(tf.abs(df)), loc, "
      "tf.nn.softplus(scale)) instead.",
      warn_once=True)
  def __init__(self,
               df,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="StudentTWithAbsDfSoftplusScale"):
    # [FUNCTION DOC]
    # Constructor with the same arguments as StudentT (df, loc, scale, validate_args,
    # allow_nan_stats, name). Passes floor of absolute df, unchanged loc and softplus of scale to
    # the StudentT constructor inside a name scope, then overwrites the stored parameters with the
    # raw arguments originally passed. Note a df whose absolute value floors to zero would be
    # invalid. Deprecated; warns once.
    parameters = dict(locals())
    with ops.name_scope(name, values=[df, scale]) as name:
      super(StudentTWithAbsDfSoftplusScale, self).__init__(
          df=math_ops.floor(math_ops.abs(df)),
          loc=loc,
          scale=nn.softplus(scale, name="softplus_scale"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters
