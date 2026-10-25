# [MODULE DOC]
# Defines the deprecated Uniform distribution (continuous uniform on the half-open interval from
# low to high, with broadcastable parameters). Exposes the Uniform class, which is exported as the
# version-one distributions Uniform symbol. Import-time side effect: the export registration only.
# Part of the legacy distributions package, superseded by TensorFlow Probability.
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
"""The Uniform distribution class."""

import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["distributions.Uniform"])
class Uniform(distribution.Distribution):
  # [CLASS DOC]
  # Batch of scalar-event uniform distributions parameterized by tensors low and high, which must
  # be broadcast-compatible with low less than high. State: the low and high tensors (identity
  # copies). Fully reparameterized. Implements batch/event shapes, sampling, density, cdf,
  # entropy, mean, variance and standard deviation. Exported under the v1 distributions namespace.
  """Uniform distribution with `low` and `high` parameters.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; a, b) = I[a <= x < b] / Z
  Z = b - a
  ```

  where

  - `low = a`,
  - `high = b`,
  - `Z` is the normalizing constant, and
  - `I[predicate]` is the [indicator function](
    https://en.wikipedia.org/wiki/Indicator_function) for `predicate`.

  The parameters `low` and `high` must be shaped in a way that supports
  broadcasting (e.g., `high - low` is a valid operation).

  #### Examples

  ```python
  # Without broadcasting:
  u1 = Uniform(low=3.0, high=4.0)  # a single uniform distribution [3, 4]
  u2 = Uniform(low=[1.0, 2.0],
               high=[3.0, 4.0])  # 2 distributions [1, 3], [2, 4]
  u3 = Uniform(low=[[1.0, 2.0],
                    [3.0, 4.0]],
               high=[[1.5, 2.5],
                     [3.5, 4.5]])  # 4 distributions
  ```

  ```python
  # With broadcasting:
  u1 = Uniform(low=3.0, high=[5.0, 6.0, 7.0])  # 3 distributions
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
               low=0.,
               high=1.,
               validate_args=False,
               allow_nan_stats=True,
               name="Uniform"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says an invalid-argument error is raised when low >= high and
    # validate_args is false; the code adds the less-than assertion only when validate_args is
    # true, so with false nothing is checked and invalid inputs silently give wrong results.
    # Constructs the distribution. Inputs: low and high (floating tensors or values, defaults zero
    # and one), validate_args (default false), allow_nan_stats (default true), name (default
    # Uniform). Stores identity copies of low and high (under a control dependency on the
    # assertion when validating), checks both have the same float dtype (raises a type error
    # otherwise), then calls the parent constructor with fully reparameterized type and the two
    # tensors as graph parents. Deprecated; emits a one-time warning. Returns nothing.
    """Initialize a batch of Uniform distributions.

    Args:
      low: Floating point tensor, lower boundary of the output interval. Must
        have `low < high`.
      high: Floating point tensor, upper boundary of the output interval. Must
        have `low < high`.
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
      InvalidArgumentError: if `low >= high` and `validate_args=False`.
    """
    parameters = dict(locals())
    with ops.name_scope(name, values=[low, high]) as name:
      with ops.control_dependencies([
          check_ops.assert_less(
              low, high, message="uniform not defined when low >= high.")
      ] if validate_args else []):
        self._low = array_ops.identity(low, name="low")
        self._high = array_ops.identity(high, name="high")
        check_ops.assert_same_float_dtype([self._low, self._high])
    super(Uniform, self).__init__(
        dtype=self._low.dtype,
        reparameterization_type=distribution.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._low,
                       self._high],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    # [FUNCTION DOC]
    # HELPER: supplies parameter shapes for the parent class's parameter-shape utilities.
    # Input: desired sample shape (vector). Returns a mapping from the names low and high to
    # the sample shape converted to an int32 tensor (the same shape for both).
    return dict(
        zip(("low", "high"),
            ([ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def low(self):
    # [FUNCTION DOC]
    # Read-only property returning the lower boundary tensor of the interval.
    """Lower boundary of the output interval."""
    return self._low

  @property
  def high(self):
    # [FUNCTION DOC]
    # Read-only property returning the upper boundary tensor of the interval.
    """Upper boundary of the output interval."""
    return self._high

  def range(self, name="range"):
    # [FUNCTION DOC]
    # Returns the interval width (high minus low), broadcast over the batch. Input: optional name
    # for the name scope (default range). Creates a subtraction op.
    """`high - low`."""
    with self._name_scope(name):
      return self.high - self.low

  def _batch_shape_tensor(self):
    # [FUNCTION DOC]
    # Dynamic batch shape: the broadcast of the dynamic shapes of low and high. Returns an
    # integer vector tensor.
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.low),
        array_ops.shape(self.high))

  def _batch_shape(self):
    # [FUNCTION DOC]
    # Static batch shape: the broadcast of the static shapes of low and high. Raises a value error
    # if they are incompatible. Returns a static shape object.
    return array_ops.broadcast_static_shape(
        self.low.get_shape(),
        self.high.get_shape())

  def _event_shape_tensor(self):
    # [FUNCTION DOC]
    # Dynamic event shape: always an empty int32 vector since the event is scalar.
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    # [FUNCTION DOC]
    # Static event shape: always the scalar (empty) shape.
    return tensor_shape.TensorShape([])

  def _sample_n(self, n, seed=None):
    # [FUNCTION DOC]
    # Draws n samples per batch member. Inputs: n (count), seed (optional random seed). Generates
    # standard uniform values of shape n plus batch shape and rescales them as low plus width times
    # the draw. Returns the samples tensor. Uses randomness.
    shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
    samples = random_ops.random_uniform(shape=shape,
                                        dtype=self.dtype,
                                        seed=seed)
    return self.low + self.range() * samples

  def _prob(self, x):
    # [FUNCTION DOC]
    # Density at x: one over the width inside the interval [low, high) and zero outside; not-a-number
    # inputs propagate. x is broadcast to the batch shape first. Returns a tensor.
    broadcasted_x = x * array_ops.ones(
        self.batch_shape_tensor(), dtype=x.dtype)
    return array_ops.where_v2(
        math_ops.is_nan(broadcasted_x), broadcasted_x,
        array_ops.where_v2(
            math_ops.logical_or(broadcasted_x < self.low,
                                broadcasted_x >= self.high),
            array_ops.zeros_like(broadcasted_x),
            array_ops.ones_like(broadcasted_x) / self.range()))

  def _cdf(self, x):
    # [FUNCTION DOC]
    # Cumulative distribution at x: zero below low, one at or above high, and (x minus low)
    # divided by the width in between. Broadcasts x with the batch shape. Returns a tensor.
    broadcast_shape = array_ops.broadcast_dynamic_shape(
        array_ops.shape(x), self.batch_shape_tensor())
    zeros = array_ops.zeros(broadcast_shape, dtype=self.dtype)
    ones = array_ops.ones(broadcast_shape, dtype=self.dtype)
    broadcasted_x = x * ones
    result_if_not_big = array_ops.where_v2(
        x < self.low, zeros, (broadcasted_x - self.low) / self.range())
    return array_ops.where_v2(x >= self.high, ones, result_if_not_big)

  def _entropy(self):
    # [FUNCTION DOC]
    # Differential entropy: the natural log of the interval width. Returns a tensor.
    return math_ops.log(self.range())

  def _mean(self):
    # [FUNCTION DOC]
    # Mean: midpoint of low and high. Returns a tensor.
    return (self.low + self.high) / 2.

  def _variance(self):
    # [FUNCTION DOC]
    # Variance: width squared divided by twelve. Returns a tensor.
    return math_ops.square(self.range()) / 12.

  def _stddev(self):
    # [FUNCTION DOC]
    # Standard deviation: width divided by the square root of twelve. Returns a tensor.
    return self.range() / math.sqrt(12.)
