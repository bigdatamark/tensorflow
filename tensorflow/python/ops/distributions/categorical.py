# [MODULE DOC]
# Defines the deprecated Categorical distribution (a discrete distribution over class indices
# parameterized by logits or probabilities), a private broadcasting helper, and registers the
# analytic KL divergence between two Categorical distributions with the KL registry (import-time
# side effect). The class is exported to the public v1 distributions namespace. Builds on the
# generic distribution base class and distribution utility functions.
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
"""The Categorical distribution class."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


def _broadcast_cat_event_and_params(event, params, base_dtype):
  # [FUNCTION DOC]
  # HELPER: Shared by the Categorical cdf and log-probability computations.
  # Inputs: event (tensor of class indices, integer or floating dtype), params (tensor of
  # logits or probabilities whose last dimension indexes classes) and base_dtype (the
  # distribution's dtype, used only in the error message). Integer events are kept, floating
  # events are cast to 32-bit integers, other dtypes raise a type error. If the batch shape of
  # params and the shape of event are not both statically known and equal, both are broadcast
  # against each other (params gains the event's shape in front of its class dimension; event
  # is multiplied by ones of the params' batch shape) and the static shape of event is
  # updated when the params rank is known. Returns a pair: the broadcast event and params.
  """Broadcasts the event or distribution parameters."""
  if event.dtype.is_integer:
    pass
  elif event.dtype.is_floating:
    # When `validate_args=True` we've already ensured int/float casting
    # is closed.
    event = math_ops.cast(event, dtype=dtypes.int32)
  else:
    raise TypeError("`value` should have integer `dtype` or "
                    "`self.dtype` ({})".format(base_dtype))
  shape_known_statically = (
      params.shape.ndims is not None and
      params.shape[:-1].is_fully_defined() and
      event.shape.is_fully_defined())
  if not shape_known_statically or params.shape[:-1] != event.shape:
    params *= array_ops.ones_like(event[..., array_ops.newaxis],
                                  dtype=params.dtype)
    params_shape = array_ops.shape(params)[:-1]
    event *= array_ops.ones(params_shape, dtype=event.dtype)
    if params.shape.ndims is not None:
      event.set_shape(tensor_shape.TensorShape(params.shape[:-1]))

  return event, params


@tf_export(v1=["distributions.Categorical"])
class Categorical(distribution.Distribution):
  # [CLASS DOC]
  # KNOWN ISSUE: The docstring says the support is the integers zero through K, but with K
  # classes the support is zero through K minus one.
  # Deprecated discrete distribution over K class indices, parameterized by a logits tensor or
  # a probabilities tensor whose last dimension is the class axis and whose leading dimensions
  # form the batch. Stores logits, probs, batch rank, event size and batch shape tensors
  # computed at construction. Events are scalars of the configured integer or float dtype;
  # samples are not reparameterized. Overrides the underscore hooks for shapes, sampling, cdf,
  # log probability, entropy and mode. Construction emits a one-time deprecation warning.
  """Categorical distribution.

  The Categorical distribution is parameterized by either probabilities or
  log-probabilities of a set of `K` classes. It is defined over the integers
  `{0, 1, ..., K}`.

  The Categorical distribution is closely related to the `OneHotCategorical` and
  `Multinomial` distributions.  The Categorical distribution can be intuited as
  generating samples according to `argmax{ OneHotCategorical(probs) }` itself
  being identical to `argmax{ Multinomial(probs, total_count=1) }`.

  #### Mathematical Details

  The probability mass function (pmf) is,

  ```none
  pmf(k; pi) = prod_j pi_j**[k == j]
  ```

  #### Pitfalls

  The number of classes, `K`, must not exceed:
  - the largest integer representable by `self.dtype`, i.e.,
    `2**(mantissa_bits+1)` (IEEE 754),
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

  Creates a 3-class distribution with the 2nd class being most likely.

  ```python
  dist = Categorical(probs=[0.1, 0.5, 0.4])
  n = 1e4
  empirical_prob = tf.cast(
      tf.histogram_fixed_width(
        dist.sample(int(n)),
        [0., 2],
        nbins=3),
      dtype=tf.float32) / n
  # ==> array([ 0.1005,  0.5037,  0.3958], dtype=float32)
  ```

  Creates a 3-class distribution with the 2nd class being most likely.
  Parameterized by [logits](https://en.wikipedia.org/wiki/Logit) rather than
  probabilities.

  ```python
  dist = Categorical(logits=np.log([0.1, 0.5, 0.4])
  n = 1e4
  empirical_prob = tf.cast(
      tf.histogram_fixed_width(
        dist.sample(int(n)),
        [0., 2],
        nbins=3),
      dtype=tf.float32) / n
  # ==> array([0.1045,  0.5047, 0.3908], dtype=float32)
  ```

  Creates a 3-class distribution with the 3rd class being most likely.
  The distribution functions can be evaluated on counts.

  ```python
  # counts is a scalar.
  p = [0.1, 0.4, 0.5]
  dist = Categorical(probs=p)
  dist.prob(0)  # Shape []

  # p will be broadcast to [[0.1, 0.4, 0.5], [0.1, 0.4, 0.5]] to match counts.
  counts = [1, 0]
  dist.prob(counts)  # Shape [2]

  # p will be broadcast to shape [3, 5, 7, 3] to match counts.
  counts = [[...]] # Shape [5, 7, 3]
  dist.prob(counts)  # Shape [5, 7, 3]
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
  def __init__(
      self,
      logits=None,
      probs=None,
      dtype=dtypes.int32,
      validate_args=False,
      allow_nan_stats=True,
      name="Categorical"):
    # [FUNCTION DOC]
    # Constructs a batch of Categorical distributions. Inputs: logits or probs (exactly one
    # should be given; tensors of rank at least one with classes on the last axis; the other
    # is derived), dtype (event sample type, default 32-bit integer), validate_args (adds
    # parameter checks, including a check that the class count fits the event dtype),
    # allow_nan_stats and a name for the op scope. Converts the parameters via the shared
    # logits/probs utility (which raises a value error if both or neither are provided),
    # records the batch rank, event size (number of classes) and batch shape as tensors using
    # static information when available and dynamic shape ops otherwise, and calls the base
    # constructor with non-reparameterized sampling. Returns nothing; emits a deprecation warning.
    """Initialize Categorical distributions using class log-probabilities.

    Args:
      logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities
        of a set of Categorical distributions. The first `N - 1` dimensions
        index into a batch of independent distributions and the last dimension
        represents a vector of logits for each class. Only one of `logits` or
        `probs` should be passed in.
      probs: An N-D `Tensor`, `N >= 1`, representing the probabilities
        of a set of Categorical distributions. The first `N - 1` dimensions
        index into a batch of independent distributions and the last dimension
        represents a vector of probabilities for each class. Only one of
        `logits` or `probs` should be passed in.
      dtype: The type of the event samples (default: int32).
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
    with ops.name_scope(name, values=[logits, probs]) as name:
      self._logits, self._probs = distribution_util.get_logits_and_probs(
          logits=logits,
          probs=probs,
          validate_args=validate_args,
          multidimensional=True,
          name=name)

      if validate_args:
        self._logits = distribution_util.embed_check_categorical_event_shape(
            self._logits)

      logits_shape_static = self._logits.get_shape().with_rank_at_least(1)
      if logits_shape_static.ndims is not None:
        self._batch_rank = ops.convert_to_tensor(
            logits_shape_static.ndims - 1,
            dtype=dtypes.int32,
            name="batch_rank")
      else:
        with ops.name_scope(name="batch_rank"):
          self._batch_rank = array_ops.rank(self._logits) - 1

      logits_shape = array_ops.shape(self._logits, name="logits_shape")
      if tensor_shape.dimension_value(logits_shape_static[-1]) is not None:
        self._event_size = ops.convert_to_tensor(
            logits_shape_static.dims[-1].value,
            dtype=dtypes.int32,
            name="event_size")
      else:
        with ops.name_scope(name="event_size"):
          self._event_size = logits_shape[self._batch_rank]

      if logits_shape_static[:-1].is_fully_defined():
        self._batch_shape_val = constant_op.constant(
            logits_shape_static[:-1].as_list(),
            dtype=dtypes.int32,
            name="batch_shape")
      else:
        with ops.name_scope(name="batch_shape"):
          self._batch_shape_val = logits_shape[:-1]
    super(Categorical, self).__init__(
        dtype=dtype,
        reparameterization_type=distribution.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._logits,
                       self._probs],
        name=name)

  @property
  def event_size(self):
    # [FUNCTION DOC]
    # Read-only property returning the scalar 32-bit integer tensor holding the number of classes.
    """Scalar `int32` tensor: the number of classes."""
    return self._event_size

  @property
  def logits(self):
    # [FUNCTION DOC]
    # Read-only property returning the logits tensor (given directly or derived from probs).
    """Vector of coordinatewise logits."""
    return self._logits

  @property
  def probs(self):
    # [FUNCTION DOC]
    # Read-only property returning the probabilities tensor (given directly or derived from logits).
    """Vector of coordinatewise probabilities."""
    return self._probs

  def _batch_shape_tensor(self):
    # [FUNCTION DOC]
    # Returns the dynamic batch shape as an integer tensor (a copy of the stored batch shape,
    # i.e. all logits dimensions except the class axis).
    return array_ops.identity(self._batch_shape_val)

  def _batch_shape(self):
    # [FUNCTION DOC]
    # Returns the static batch shape: the static shape of the logits without its last dimension.
    return self.logits.get_shape()[:-1]

  def _event_shape_tensor(self):
    # [FUNCTION DOC]
    # Returns the dynamic event shape: an empty 32-bit integer vector, since events are scalars.
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    # [FUNCTION DOC]
    # Returns the static event shape: a fully known empty (scalar) shape.
    return tensor_shape.TensorShape([])

  def _sample_n(self, n, seed=None):
    # [FUNCTION DOC]
    # Draws n samples. Inputs: n (scalar sample count) and an optional integer seed. Flattens the
    # logits to two dimensions if needed, draws from the multinomial sampler (64-bit output when
    # the distribution dtype is wider than four bytes, else 32-bit), transposes and reshapes the
    # draws to the sample count followed by the batch shape, and casts to the distribution dtype.
    # Consumes randomness; returns the sample tensor.
    if self.logits.get_shape().ndims == 2:
      logits_2d = self.logits
    else:
      logits_2d = array_ops.reshape(self.logits, [-1, self.event_size])
    sample_dtype = dtypes.int64 if self.dtype.size > 4 else dtypes.int32
    draws = random_ops.multinomial(
        logits_2d, n, seed=seed, output_dtype=sample_dtype)
    draws = array_ops.reshape(
        array_ops.transpose(draws),
        array_ops.concat([[n], self.batch_shape_tensor()], 0))
    return math_ops.cast(draws, self.dtype)

  def _cdf(self, k):
    # [FUNCTION DOC]
    # Cumulative distribution function. Input: k, class indices (integer or float values
    # convertible to a tensor). With validation on, checks the values can be cast to integers
    # without loss. Broadcasts k with the probabilities, flattens the batch, and sums the
    # probabilities of all classes strictly below each k (a sequence mask selects them), then
    # reshapes to the shape of k. Returns a tensor of probabilities. Note this is the
    # probability of values strictly less than k, not less than or equal to k.
    # KNOWN ISSUE: Because the mask excludes class k itself, this returns P(X < k), which differs
    # from the conventional definition of a cdf, P(X <= k).
    k = ops.convert_to_tensor(k, name="k")
    if self.validate_args:
      k = distribution_util.embed_check_integer_casting_closed(
          k, target_dtype=dtypes.int32)

    k, probs = _broadcast_cat_event_and_params(
        k, self.probs, base_dtype=self.dtype.base_dtype)

    # batch-flatten everything in order to use `sequence_mask()`.
    batch_flattened_probs = array_ops.reshape(probs,
                                              (-1, self._event_size))
    batch_flattened_k = array_ops.reshape(k, [-1])

    to_sum_over = array_ops.where(
        array_ops.sequence_mask(batch_flattened_k, self._event_size),
        batch_flattened_probs,
        array_ops.zeros_like(batch_flattened_probs))
    batch_flattened_cdf = math_ops.reduce_sum(to_sum_over, axis=-1)
    # Reshape back to the shape of the argument.
    return array_ops.reshape(batch_flattened_cdf, array_ops.shape(k))

  def _log_prob(self, k):
    # [FUNCTION DOC]
    # Log probability mass. Input: k, class indices (integer or float values convertible to a
    # tensor; validated for lossless integer casting when validate_args is set). Broadcasts k with
    # the logits and returns the negative sparse softmax cross entropy, which equals the log
    # probability of each class index. Out-of-range indices are not checked here.
    k = ops.convert_to_tensor(k, name="k")
    if self.validate_args:
      k = distribution_util.embed_check_integer_casting_closed(
          k, target_dtype=dtypes.int32)
    k, logits = _broadcast_cat_event_and_params(
        k, self.logits, base_dtype=self.dtype.base_dtype)

    # pylint: disable=invalid-unary-operand-type
    return -nn_ops.sparse_softmax_cross_entropy_with_logits(
        labels=k,
        logits=logits)

  def _entropy(self):
    # [FUNCTION DOC]
    # Entropy per batch member: negative sum over classes of log-softmax of the logits times the
    # probabilities. Returns a tensor of the batch shape.
    return -math_ops.reduce_sum(
        nn_ops.log_softmax(self.logits) * self.probs, axis=-1)

  def _mode(self):
    # [FUNCTION DOC]
    # Mode per batch member: the index of the largest logit along the class axis, cast to the
    # distribution dtype with its static shape set to the batch shape.
    ret = math_ops.argmax(self.logits, axis=self._batch_rank)
    ret = math_ops.cast(ret, self.dtype)
    ret.set_shape(self.batch_shape)
    return ret


@kullback_leibler.RegisterKL(Categorical, Categorical)
def _kl_categorical_categorical(a, b, name=None):
  # [FUNCTION DOC]
  # HELPER: KL divergence implementation registered for the pair (Categorical, Categorical).
  # Inputs: a and b, Categorical distributions with matching class counts and broadcastable batch
  # shapes, and an optional op name. Returns the batchwise KL divergence of a from b: the sum over
  # classes of a's probabilities times the difference of the two log-softmaxes of the logits.
  """Calculate the batched KL divergence KL(a || b) with a and b Categorical.

  Args:
    a: instance of a Categorical distribution object.
    b: instance of a Categorical distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_categorical_categorical".

  Returns:
    Batchwise KL(a || b)
  """
  with ops.name_scope(name, "kl_categorical_categorical",
                      values=[a.logits, b.logits]):
    # sum(probs log(probs / (1 - probs)))
    delta_log_probs1 = (nn_ops.log_softmax(a.logits) -
                        nn_ops.log_softmax(b.logits))
    return math_ops.reduce_sum(nn_ops.softmax(a.logits) * delta_log_probs1,
                               axis=-1)
