# [MODULE DOC]
# Defines the legacy (deprecated) Bernoulli distribution class and registers the closed-form KL
# divergence between two Bernoulli distributions with the KL registry. Exported through the v1
# API only. Import-time side effects: class export decorator and KL registration.
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
"""The Bernoulli distribution class."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["distributions.Bernoulli"])
class Bernoulli(distribution.Distribution):
  # [CLASS DOC]
  # Bernoulli distribution over {0, 1} parameterized by either logits (log-odds) or probs
  # (probability of 1); both forms are stored as tensors and exposed as properties. Subclass of the
  # base Distribution; samples are not reparameterizable, events are scalars and batch shape is the
  # shape of the parameters. Exported as a v1-only API (distributions.Bernoulli) and deprecated.
  # Statistics provided: mean, variance, mode, entropy, log probability and sampling.
  """Bernoulli distribution.

  The Bernoulli distribution with `probs` parameter, i.e., the probability of a
  `1` outcome (vs a `0` outcome).
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
               logits=None,
               probs=None,
               dtype=dtypes.int32,
               validate_args=False,
               allow_nan_stats=True,
               name="Bernoulli"):
    # [FUNCTION DOC]
    # Constructor (deprecated since 2019-01-01, warns once; the library moved to TensorFlow
    # Probability). Builds a Bernoulli distribution from either logits or probs (exactly one should
    # be given). Inputs: logits and probs are optional tensors (log-odds, probability of 1);
    # dtype is the sample type (default int32); validate_args enables runtime checks (default
    # false); allow_nan_stats chooses NaN versus raising for undefined statistics (default true);
    # name is the op-name scope (default "Bernoulli"). Converts the given parameter to the other via
    # a shared utility, then calls the base constructor with a not-reparameterized marker and the
    # parameters captured from the local arguments. Raises a value error if both or neither of
    # logits and probs are given (raised by the conversion utility).
    """Construct Bernoulli distributions.

    Args:
      logits: An N-D `Tensor` representing the log-odds of a `1` event. Each
        entry in the `Tensor` parametrizes an independent Bernoulli distribution
        where the probability of an event is sigmoid(logits). Only one of
        `logits` or `probs` should be passed in.
      probs: An N-D `Tensor` representing the probability of a `1`
        event. Each entry in the `Tensor` parameterizes an independent
        Bernoulli distribution. Only one of `logits` or `probs` should be passed
        in.
      dtype: The type of the event samples. Default: `int32`.
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
      ValueError: If p and logits are passed, or if neither are passed.
    """
    parameters = dict(locals())
    with ops.name_scope(name) as name:
      self._logits, self._probs = distribution_util.get_logits_and_probs(
          logits=logits,
          probs=probs,
          validate_args=validate_args,
          name=name)
    super(Bernoulli, self).__init__(
        dtype=dtype,
        reparameterization_type=distribution.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._logits, self._probs],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    # [FUNCTION DOC]
    # HELPER: Static helper used by the generic distribution machinery to describe parameter shapes.
    # Input: sample_shape, a shape (list or tensor of integers). Returns a mapping with a single
    # entry, "logits", whose value is the sample shape converted to an int32 tensor. The shape of
    # the logits parameter therefore equals the sample shape. No other side effects.
    return {"logits": ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)}

  @property
  def logits(self):
    # [FUNCTION DOC]
    # Property returning the log-odds of a 1 outcome as stored by the constructor (a tensor).
    # No side effects.
    """Log-odds of a `1` outcome (vs `0`)."""
    return self._logits

  @property
  def probs(self):
    # [FUNCTION DOC]
    # Property returning the probability of a 1 outcome as stored by the constructor (a tensor).
    # No side effects.
    """Probability of a `1` outcome (vs `0`)."""
    return self._probs

  def _batch_shape_tensor(self):
    # [FUNCTION DOC]
    # Returns the dynamic batch shape as a tensor, i.e. the shape of the logits tensor.
    # No arguments and no side effects other than creating a shape op.
    return array_ops.shape(self._logits)

  def _batch_shape(self):
    # [FUNCTION DOC]
    # Returns the static batch shape, taken from the static shape of the logits tensor.
    # No arguments and no side effects.
    return self._logits.get_shape()

  def _event_shape_tensor(self):
    # [FUNCTION DOC]
    # Returns the dynamic event shape as an empty int32 constant tensor (scalar events).
    # No arguments, no side effects other than creating a constant op.
    return array_ops.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    # [FUNCTION DOC]
    # Returns the static event shape: an empty (scalar) tensor shape, since a Bernoulli event is a
    # scalar. No arguments, no side effects.
    return tensor_shape.TensorShape([])

  def _sample_n(self, n, seed=None):
    # [FUNCTION DOC]
    # Draws n independent samples. The sample shape is n followed by the batch shape; a uniform
    # random value is drawn for each element and compared to probs (sample is 1 when below probs).
    # Inputs: n is the number of samples (integer or scalar tensor); seed is an optional random seed.
    # Returns a tensor of the distribution dtype. Consumes randomness; no other side effects.
    new_shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
    uniform = random_ops.random_uniform(
        new_shape, seed=seed, dtype=self.probs.dtype)
    sample = math_ops.less(uniform, self.probs)
    return math_ops.cast(sample, self.dtype)

  def _log_prob(self, event):
    # [FUNCTION DOC]
    # Returns the log-probability of the given event values (0 or 1, any numeric type).
    # If validate_args is set, first adds a check that event is integer-valued (castable to a
    # boolean) and embeds it as a dependency. The event is cast to the logits dtype; when the static
    # shapes of event and logits are not fully defined and equal, both are explicitly broadcast.
    # The result is the negative sigmoid cross entropy with logits, a tensor of log-probabilities.
    # Raises a runtime/graph error only if the validation check fails.
    if self.validate_args:
      event = distribution_util.embed_check_integer_casting_closed(
          event, target_dtype=dtypes.bool)

    # TODO(jaana): The current sigmoid_cross_entropy_with_logits has
    # inconsistent behavior for logits = inf/-inf.
    event = math_ops.cast(event, self.logits.dtype)
    logits = self.logits
    # sigmoid_cross_entropy_with_logits doesn't broadcast shape,
    # so we do this here.

    def _broadcast(logits, event):
      # [FUNCTION DOC]
      # HELPER: Nested helper of the log-probability computation that makes event and logits shapes match.
      # Multiplies each of event and logits by an all-ones tensor shaped like the other one, so both
      # broadcast to the common shape. Returns a pair (broadcast logits, broadcast event). No side effects.
      return (array_ops.ones_like(event) * logits,
              array_ops.ones_like(logits) * event)

    if not (event.get_shape().is_fully_defined() and
            logits.get_shape().is_fully_defined() and
            event.get_shape() == logits.get_shape()):
      logits, event = _broadcast(logits, event)
    return -nn.sigmoid_cross_entropy_with_logits(labels=event, logits=logits)

  def _entropy(self):
    # [FUNCTION DOC]
    # Returns the entropy of each batch member computed from logits: -logits * (sigmoid(logits) - 1)
    # plus softplus(-logits), a numerically stable form of the binary entropy. No arguments, no
    # side effects.
    return (-self.logits * (math_ops.sigmoid(self.logits) - 1) +  # pylint: disable=invalid-unary-operand-type
            nn.softplus(-self.logits))  # pylint: disable=invalid-unary-operand-type

  def _mean(self):
    # [FUNCTION DOC]
    # Returns the mean of each batch member, which equals probs (via an identity op so a new tensor
    # is returned). No arguments and no side effects.
    return array_ops.identity(self.probs)

  def _variance(self):
    # [FUNCTION DOC]
    # Returns the variance of each batch member, mean * (1 - probs), i.e. p * (1 - p).
    # No arguments and no side effects.
    return self._mean() * (1. - self.probs)

  def _mode(self):
    # [FUNCTION DOC]
    # Returns the mode: 1 where probs exceeds 0.5, else 0, cast to the distribution dtype.
    # No arguments, no side effects. Ties at exactly 0.5 give 0.
    """Returns `1` if `prob > 0.5` and `0` otherwise."""
    return math_ops.cast(self.probs > 0.5, self.dtype)


@kullback_leibler.RegisterKL(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(a, b, name=None):
  # [FUNCTION DOC]
  # HELPER: Registered KL-divergence implementation used by the generic KL dispatch machinery.
  # Computes the element-wise KL divergence between two Bernoulli distributions a and b from their
  # logits, using softplus differences weighted by sigmoid(a.logits) and sigmoid(-a.logits).
  # Inputs: a and b are Bernoulli instances; name is an optional op-name scope (default
  # "kl_bernoulli_bernoulli"). Returns a tensor of batchwise divergences. Registered at import
  # time for the pair (Bernoulli, Bernoulli). Creates graph ops only; raises nothing itself.
  """Calculate the batched KL divergence KL(a || b) with a and b Bernoulli.

  Args:
    a: instance of a Bernoulli distribution object.
    b: instance of a Bernoulli distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_bernoulli_bernoulli".

  Returns:
    Batchwise KL(a || b)
  """
  with ops.name_scope(name, "kl_bernoulli_bernoulli",
                      values=[a.logits, b.logits]):
    delta_probs0 = nn.softplus(-b.logits) - nn.softplus(-a.logits)
    delta_probs1 = nn.softplus(b.logits) - nn.softplus(a.logits)
    return (math_ops.sigmoid(a.logits) * delta_probs0
            + math_ops.sigmoid(-a.logits) * delta_probs1)
