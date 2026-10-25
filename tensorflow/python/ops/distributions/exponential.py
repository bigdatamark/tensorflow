# [MODULE DOC]
# Defines the deprecated (v1 only) Exponential distribution, implemented as a Gamma distribution
# with concentration fixed at one, and a variant whose rate is passed through a softplus
# transform. Exposes the classes Exponential and ExponentialWithSoftplusRate via the module
# export list; only Exponential is registered with the public API export decorator (import-time
# side effect). Depends on the Gamma distribution module for most behavior.
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
"""The Exponential distribution class."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import gamma
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


__all__ = [
    "Exponential",
    "ExponentialWithSoftplusRate",
]


@tf_export(v1=["distributions.Exponential"])
class Exponential(gamma.Gamma):
  # [CLASS DOC]
  # Exponential distribution over positive reals parameterized by a positive rate (inverse
  # scale; mean is one over rate). Subclass of Gamma that always uses a concentration of one.
  # Holds the rate tensor; overrides the parameter-shape helper, the log survival function and
  # sampling (inverse-transform of uniform variates) while inheriting the other statistics from
  # Gamma. Created from a rate array-like; deprecated in favor of the separate probability library.
  """Exponential distribution.

  The Exponential distribution is parameterized by an event `rate` parameter.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; lambda, x > 0) = exp(-lambda x) / Z
  Z = 1 / lambda
  ```

  where `rate = lambda` and `Z` is the normalizaing constant.

  The Exponential distribution is a special case of the Gamma distribution,
  i.e.,

  ```python
  Exponential(rate) = Gamma(concentration=1., rate)
  ```

  The Exponential distribution uses a `rate` parameter, or "inverse scale",
  which can be intuited as,

  ```none
  X ~ Exponential(rate=1)
  Y = X / rate
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
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name="Exponential"):
    # [FUNCTION DOC]
    # Constructor. Converts rate to a tensor inside a name scope, then initializes the parent
    # Gamma with a scalar concentration of one (same dtype as rate) and that rate. Afterwards it
    # overwrites the stored parameters with this class's own constructor arguments and appends
    # the rate tensor to the graph parents list.
    # Inputs: rate (positive float array-like, equal to one over the mean); validate_args
    # (default false) enables parameter checks in the parent; allow_nan_stats (default true);
    # name (default "Exponential"). Outputs: none. Side effects: ops created in a name scope;
    # deprecation decorator logs a one-time warning. Invalid rates raise assertion errors only
    # when validate_args is true.
    """Construct Exponential distribution with parameter `rate`.

    Args:
      rate: Floating point tensor, equivalent to `1 / mean`. Must contain only
        positive values.
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
    # Even though all statistics of are defined for valid inputs, this is not
    # true in the parent class "Gamma."  Therefore, passing
    # allow_nan_stats=True
    # through to the parent class results in unnecessary asserts.
    with ops.name_scope(name, values=[rate]) as name:
      self._rate = ops.convert_to_tensor(rate, name="rate")
    super(Exponential, self).__init__(
        concentration=array_ops.ones([], dtype=self._rate.dtype),
        rate=self._rate,
        allow_nan_stats=allow_nan_stats,
        validate_args=validate_args,
        name=name)
    self._parameters = parameters
    self._graph_parents += [self._rate]

  @staticmethod
  def _param_shapes(sample_shape):
    # [FUNCTION DOC]
    # HELPER: Static helper used by the base class parameter-shape utilities.
    # Input: sample_shape, a shape given as a list or tensor of integers. Returns a mapping
    # with a single entry, rate, whose value is the sample shape converted to a 32-bit integer
    # tensor (the rate has the same shape as the samples' batch).
    return {"rate": ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)}

  @property
  def rate(self):
    # [FUNCTION DOC]
    # Read-only property returning the rate tensor (inverse scale) given at construction.
    return self._rate

  def _log_survival_function(self, value):
    # [FUNCTION DOC]
    # HELPER: Base-class hook for the public log survival function method.
    # Input: value, tensor of points (non-negative expected). Returns the log probability
    # density at value minus the log of the rate, which equals minus rate times value (the
    # exact log of the survival function) for non-negative values. No validation is done here.
    return self._log_prob(value) - math_ops.log(self._rate)

  def _sample_n(self, n, seed=None):
    # [FUNCTION DOC]
    # HELPER: Base-class hook used by the public sample method.
    # Draws n samples by inverse-transform sampling: uniform variates in the open interval
    # between the smallest positive normal number and one, then minus log of the uniform divided
    # by the rate. Inputs: n number of samples; seed optional random seed. Returns a tensor of
    # shape [n] plus the rate shape with the distribution dtype. Uses the random generator.
    shape = array_ops.concat([[n], array_ops.shape(self._rate)], 0)
    # Uniform variates must be sampled from the open-interval `(0, 1)` rather
    # than `[0, 1)`. To do so, we use `np.finfo(self.dtype.as_numpy_dtype).tiny`
    # because it is the smallest, positive, "normal" number. A "normal" number
    # is such that the mantissa has an implicit leading 1. Normal, positive
    # numbers x, y have the reasonable property that, `x + y >= max(x, y)`. In
    # this case, a subnormal number (i.e., np.nextafter) can cause us to sample
    # 0.
    sampled = random_ops.random_uniform(
        shape,
        minval=np.finfo(self.dtype.as_numpy_dtype).tiny,
        maxval=1.,
        seed=seed,
        dtype=self.dtype)
    return -math_ops.log(sampled) / self._rate


class ExponentialWithSoftplusRate(Exponential):
  # [CLASS DOC]
  # Exponential distribution whose rate parameter is the softplus transform of an unconstrained
  # input, guaranteeing a positive rate. Adds no state beyond the parent class and is not
  # exported in the public API; deprecated in favor of building an Exponential from a softplus
  # of the rate directly.
  """Exponential with softplus transform on `rate`."""

  @deprecation.deprecated(
      "2019-01-01",
      "Use `tfd.Exponential(tf.nn.softplus(rate)).",
      warn_once=True)
  def __init__(self,
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name="ExponentialWithSoftplusRate"):
    # [FUNCTION DOC]
    # Constructor. Inside a name scope, applies softplus to the raw rate input and passes the
    # result to the Exponential constructor along with the validation and NaN flags, then
    # replaces the stored parameters with this class's own constructor arguments.
    # Inputs: rate (unconstrained float array-like, mapped to a positive value); validate_args
    # (default false); allow_nan_stats (default true); name (default
    # "ExponentialWithSoftplusRate"). Outputs: none. Side effects: ops created; the
    # deprecation decorator logs a one-time warning.
    parameters = dict(locals())
    with ops.name_scope(name, values=[rate]) as name:
      super(ExponentialWithSoftplusRate, self).__init__(
          rate=nn.softplus(rate, name="softplus_rate"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters
