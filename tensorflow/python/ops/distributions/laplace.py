# [MODULE DOC]
# Defines the Laplace (double exponential) distribution of the legacy tf.distributions API (v1
# only), parameterized by location and scale, with reparameterized sampling by inverse transform
# of uniform noise. Also defines LaplaceWithSoftplusScale, which applies softplus to the scale.
# Uses the special_math module for the log CDF. Import-time effect: exports Laplace under the v1
# distributions namespace.
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
"""The Laplace distribution class."""

import math

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import special_math
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


__all__ = [
    "Laplace",
    "LaplaceWithSoftplusScale",
]


@tf_export(v1=["distributions.Laplace"])
class Laplace(distribution.Distribution):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring misspells the name as "Lpalce" in one sentence (cosmetic only).
  # Laplace distribution over the reals with location loc and positive scale. Holds both as
  # identity-wrapped tensors that broadcast together; batch shape is their broadcast shape and
  # event shape is scalar. Fully reparameterized sampling. Supports pdf, cdf, log cdf, survival
  # function, entropy, mean, median, mode and standard deviation. Deprecated.
  """The Laplace distribution with location `loc` and `scale` parameters.

  #### Mathematical details

  The probability density function (pdf) of this distribution is,

  ```none
  pdf(x; mu, sigma) = exp(-|x - mu| / sigma) / Z
  Z = 2 sigma
  ```

  where `loc = mu`, `scale = sigma`, and `Z` is the normalization constant.

  Note that the Laplace distribution can be thought of two exponential
  distributions spliced together "back-to-back."

  The Lpalce distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ Laplace(loc=0, scale=1)
  Y = loc + scale * X
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
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="Laplace"):
    # [FUNCTION DOC]
    # Constructor. loc: floating tensor of locations; scale: positive floating tensor of scales
    # (must broadcast with loc); validate_args (bool, default false) adds an assertion that scale
    # is positive; allow_nan_stats (bool, default true); name (string scope name). Stores identity
    # copies of loc and scale, checks they share a floating dtype (raises a type error otherwise)
    # and initializes the base distribution as fully reparameterized. Deprecated; warns once.
    """Construct Laplace distribution with parameters `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g., `loc / scale` is a valid operation).

    Args:
      loc: Floating point tensor which characterizes the location (center)
        of the distribution.
      scale: Positive floating point tensor which characterizes the spread of
        the distribution.
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
      TypeError: if `loc` and `scale` are of different dtype.
    """
    parameters = dict(locals())
    with ops.name_scope(name, values=[loc, scale]) as name:
      with ops.control_dependencies([check_ops.assert_positive(scale)] if
                                    validate_args else []):
        self._loc = array_ops.identity(loc, name="loc")
        self._scale = array_ops.identity(scale, name="scale")
        check_ops.assert_same_float_dtype([self._loc, self._scale])
      super(Laplace, self).__init__(
          dtype=self._loc.dtype,
          reparameterization_type=distribution.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=[self._loc, self._scale],
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    # [FUNCTION DOC]
    # HELPER: static helper for the base distribution's param_shapes machinery.
    # Returns a mapping from parameter names loc and scale to the given sample shape converted to
    # a 32-bit integer tensor (both parameters are scalar per batch member).
    return dict(
        zip(("loc", "scale"), ([ops.convert_to_tensor(
            sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def loc(self):
    # [FUNCTION DOC]
    # Property returning the location parameter tensor.
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    # [FUNCTION DOC]
    # Property returning the scale parameter tensor.
    """Distribution parameter for scale."""
    return self._scale

  def _batch_shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class batch_shape_tensor.
    # Returns the dynamic broadcast of the shapes of loc and scale as an integer tensor.
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.loc), array_ops.shape(self.scale))

  def _batch_shape(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class batch_shape.
    # Returns the static broadcast of the shapes of loc and scale; raises a value error if they
    # are not broadcast compatible.
    return array_ops.broadcast_static_shape(
        self.loc.get_shape(), self.scale.get_shape())

  def _event_shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class event_shape_tensor.
    # Returns an empty 32-bit integer vector, since draws are scalars.
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class event_shape.
    # Returns the static scalar (empty) event shape.
    return tensor_shape.TensorShape([])

  def _sample_n(self, n, seed=None):
    # [FUNCTION DOC]
    # HELPER: hook for the base class sample.
    # Draws n samples per batch member by inverse-CDF transform: samples uniform noise in the
    # open interval from just above -1 up to 1, then returns loc minus scale times the sign of
    # the noise times the log of one minus its absolute value. n is an integer count; seed is an
    # optional integer seed (default none). Returns a tensor shaped n followed by the batch shape.
    # Consumes randomness.
    shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
    # Uniform variates must be sampled from the open-interval `(-1, 1)` rather
    # than `[-1, 1)`. In the case of `(0, 1)` we'd use
    # `np.finfo(self.dtype.as_numpy_dtype).tiny` because it is the smallest,
    # positive, "normal" number. However, the concept of subnormality exists
    # only at zero; here we need the smallest usable number larger than -1,
    # i.e., `-1 + eps/2`.
    uniform_samples = random_ops.random_uniform(
        shape=shape,
        minval=np.nextafter(self.dtype.as_numpy_dtype(-1.),
                            self.dtype.as_numpy_dtype(0.)),
        maxval=1.,
        dtype=self.dtype,
        seed=seed)
    return (self.loc - self.scale * math_ops.sign(uniform_samples) *
            math_ops.log1p(-math_ops.abs(uniform_samples)))

  def _log_prob(self, x):
    # [FUNCTION DOC]
    # HELPER: hook for the base class log_prob.
    # Returns the log density at x: unnormalized log probability minus the log normalization.
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _prob(self, x):
    # [FUNCTION DOC]
    # HELPER: hook for the base class prob.
    # Returns the density at x as the exponential of the log density.
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    # [FUNCTION DOC]
    # HELPER: hook for the base class log_cdf.
    # Returns the log of the cumulative distribution at x using the numerically stable Laplace
    # log CDF helper applied to the standardized value.
    return special_math.log_cdf_laplace(self._z(x))

  def _log_survival_function(self, x):
    # [FUNCTION DOC]
    # HELPER: hook for the base class log_survival_function.
    # Returns the log of one minus the CDF at x, computed by the log CDF helper on the negated
    # standardized value (using the symmetry of the distribution).
    return special_math.log_cdf_laplace(-self._z(x))

  def _cdf(self, x):
    # [FUNCTION DOC]
    # HELPER: hook for the base class cdf.
    # Returns the cumulative distribution at x: one half plus one half times the sign of the
    # standardized value times one minus the exponential of minus its absolute value.
    z = self._z(x)
    return (0.5 + 0.5 * math_ops.sign(z) *
            (1. - math_ops.exp(-math_ops.abs(z))))

  def _log_unnormalized_prob(self, x):
    # [FUNCTION DOC]
    # HELPER: part of the log density computation.
    # Returns minus the absolute value of the standardized x.
    return -math_ops.abs(self._z(x))

  def _log_normalization(self):
    # [FUNCTION DOC]
    # HELPER: part of the log density computation.
    # Returns log 2 plus log scale (the log of the normalizing constant 2 times scale).
    return math.log(2.) + math_ops.log(self.scale)

  def _entropy(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class entropy.
    # Returns log 2 plus 1 plus log scale, with the scale first broadcast against loc so the
    # result has the full batch shape.
    # Use broadcasting rules to calculate the full broadcast scale.
    scale = self.scale + array_ops.zeros_like(self.loc)
    return math.log(2.) + 1. + math_ops.log(scale)

  def _mean(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class mean.
    # Returns loc, broadcast to the full batch shape by adding zeros shaped like scale.
    return self.loc + array_ops.zeros_like(self.scale)

  def _stddev(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class stddev.
    # Returns square root of 2 times scale, broadcast to the full batch shape against loc.
    return math.sqrt(2.) * self.scale + array_ops.zeros_like(self.loc)

  def _median(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class median. Equal to the mean (loc, broadcast).
    return self._mean()

  def _mode(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class mode. Equal to the mean (loc, broadcast).
    return self._mean()

  def _z(self, x):
    # [FUNCTION DOC]
    # HELPER: standardization used by the density and CDF methods.
    # Returns (x minus loc) divided by scale for a tensor x.
    return (x - self.loc) / self.scale


class LaplaceWithSoftplusScale(Laplace):
  # [CLASS DOC]
  # Laplace distribution whose constructor applies softplus to the scale so an unconstrained real
  # value can be supplied. Otherwise identical to Laplace. Deprecated.
  """Laplace with softplus applied to `scale`."""

  @deprecation.deprecated(
      "2019-01-01",
      "Use `tfd.Laplace(loc, tf.nn.softplus(scale)) "
      "instead.",
      warn_once=True)
  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="LaplaceWithSoftplusScale"):
    # [FUNCTION DOC]
    # Constructor with the same arguments as Laplace (loc, scale, validate_args, allow_nan_stats,
    # name). Passes loc unchanged and softplus of scale to the Laplace constructor inside a name
    # scope, then overwrites the stored parameters with the raw arguments originally passed.
    # Deprecated; warns once.
    parameters = dict(locals())
    with ops.name_scope(name, values=[loc, scale]) as name:
      super(LaplaceWithSoftplusScale, self).__init__(
          loc=loc,
          scale=nn.softplus(scale, name="softplus_scale"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters
