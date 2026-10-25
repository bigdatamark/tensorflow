# [MODULE DOC]
# Python front end for TensorFlow's stateful random number ops. It wraps the generated raw random
# ops (re-exported here through a wildcard import) with user-friendly functions: normal,
# truncated normal, uniform, shuffle, multinomial/categorical, gamma and Poisson sampling.
# Each wrapper converts arguments to tensors of the requested dtype, derives the graph-level and
# op-level seeds from the global seed machinery, calls the raw op and, where possible, sets the
# static output shape. Public functions are exported under the tf.random namespace (and legacy
# v1 names, some deprecated) and support type-based dispatch.
# Import-time side effects: registers the stateful random ops as not differentiable (except where
# a gradient is registered elsewhere) and applies the export, dispatch and deprecation decorators.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Operations for generating random numbers."""

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import shape_util

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_random_ops import *
# pylint: enable=wildcard-import

from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@tf_export("random.normal", v1=["random.normal", "random_normal"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("random_normal")
def random_normal(shape,
                  mean=0.0,
                  stddev=1.0,
                  dtype=dtypes.float32,
                  seed=None,
                  name=None):
  # [FUNCTION DOC]
  # Creates a tensor of normally distributed random values. Inputs: shape (1-D integer tensor or
  # sequence giving the output shape), mean and stddev (scalars or tensors broadcastable to each
  # other, converted to the given dtype), dtype (float type, default float32), seed (optional
  # integer combined with the global seed) and name (optional op name scope). Draws standard
  # normal samples with the raw op, scales by stddev, adds mean, and sets the static shape when
  # it can be inferred. Returns the resulting tensor. Conversion or dtype problems surface as
  # errors from tensor conversion. No state is mutated besides the random op's seed use.
  """Outputs random values from a normal distribution.

  Example that generates a new set of random values every time:

  >>> tf.random.set_seed(5);
  >>> tf.random.normal([4], 0, 1, tf.float32)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=..., dtype=float32)>

  Example that outputs a reproducible result:

  >>> tf.random.set_seed(5);
  >>> tf.random.normal([2,2], 0, 1, tf.float32, seed=1)
  <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
  array([[-1.3768897 , -0.01258316],
        [-0.169515   ,  1.0824056 ]], dtype=float32)>

  In this case, we are setting both the global and operation-level seed to
  ensure this result is reproducible.  See `tf.random.set_seed` for more
  information.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean: A Tensor or Python value of type `dtype`, broadcastable with `stddev`.
      The mean of the normal distribution.
    stddev: A Tensor or Python value of type `dtype`, broadcastable with `mean`.
      The standard deviation of the normal distribution.
    dtype: The float type of the output: `float16`, `bfloat16`, `float32`,
      `float64`. Defaults to `float32`.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      `tf.random.set_seed`
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random normal values.
  """
  with ops.name_scope(name, "random_normal", [shape, mean, stddev]) as name:
    shape_tensor = shape_util.shape_tensor(shape)
    mean_tensor = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
    stddev_tensor = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
    seed1, seed2 = random_seed.get_seed(seed)
    rnd = gen_random_ops.random_standard_normal(
        shape_tensor, dtype, seed=seed1, seed2=seed2)
    mul = rnd * stddev_tensor
    value = math_ops.add(mul, mean_tensor, name=name)
    shape_util.maybe_set_static_shape(value, shape)
    return value


ops.NotDifferentiable("RandomStandardNormal")


def parameterized_truncated_normal(shape,
                                   means=0.0,
                                   stddevs=1.0,
                                   minvals=-2.0,
                                   maxvals=2.0,
                                   dtype=dtypes.float32,
                                   seed=None,
                                   name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says values beyond 2 standard deviations are re-picked, but this
  # function takes explicit lower and upper bounds (minvals, maxvals, default -2 and 2) in
  # absolute value space and passes them straight to the raw op; the bounds are not relative to
  # the mean unless the caller makes them so. The seed argument is documented as an integer only.
  # Creates random values from a normal distribution truncated to the interval between minvals
  # and maxvals. Inputs: shape (1-D integer tensor or sequence), means, stddevs, minvals, maxvals
  # (scalars or tensors converted to dtype), dtype (default float32), seed (optional integer) and
  # name (optional). Derives both seeds from the seed machinery, calls the raw parameterized
  # truncated normal op, sets the static shape if known and returns the tensor.
  """Outputs random values from a truncated normal distribution.

  The generated values follow a normal distribution with specified mean and
  standard deviation, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    means: A 0-D Tensor or Python value of type `dtype`. The mean of the
      truncated normal distribution.
    stddevs: A 0-D Tensor or Python value of type `dtype`. The standard
      deviation of the truncated normal distribution.
    minvals: A 0-D Tensor or Python value of type `dtype`. The minimum value of
      the truncated normal distribution.
    maxvals: A 0-D Tensor or Python value of type `dtype`. The maximum value of
      the truncated normal distribution.
    dtype: The type of the output.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      `tf.random.set_seed`
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random truncated normal values.
  """
  with ops.name_scope(name, "parameterized_truncated_normal",
                      [shape, means, stddevs, minvals, maxvals]) as name:
    shape_tensor = shape_util.shape_tensor(shape)
    means_tensor = ops.convert_to_tensor(means, dtype=dtype, name="means")
    stddevs_tensor = ops.convert_to_tensor(stddevs, dtype=dtype, name="stddevs")
    minvals_tensor = ops.convert_to_tensor(minvals, dtype=dtype, name="minvals")
    maxvals_tensor = ops.convert_to_tensor(maxvals, dtype=dtype, name="maxvals")
    seed1, seed2 = random_seed.get_seed(seed)
    rnd = gen_random_ops.parameterized_truncated_normal(
        shape_tensor,
        means_tensor,
        stddevs_tensor,
        minvals_tensor,
        maxvals_tensor,
        seed=seed1,
        seed2=seed2)
    shape_util.maybe_set_static_shape(rnd, shape)
    return rnd


@tf_export("random.truncated_normal",
           v1=["random.truncated_normal", "truncated_normal"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("truncated_normal")
def truncated_normal(shape,
                     mean=0.0,
                     stddev=1.0,
                     dtype=dtypes.float32,
                     seed=None,
                     name=None):
  # [FUNCTION DOC]
  # Creates random values from a normal distribution in which samples more than two standard
  # deviations from the mean are discarded and redrawn. Inputs: shape (1-D integer tensor or
  # sequence), mean and stddev (scalars converted to dtype), dtype (floating type, default
  # float32), seed (optional integer) and name (optional). Samples a standard truncated normal
  # with the raw op, scales by stddev and shifts by mean, sets the static shape when known and
  # returns the tensor. Seeds come from the global/op seed logic.
  """Outputs random values from a truncated normal distribution.

  The values are drawn from a normal distribution with specified mean and
  standard deviation, discarding and re-drawing any samples that are more than
  two standard deviations from the mean.

  Examples:

  >>> tf.random.truncated_normal(shape=[2])
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([..., ...], dtype=float32)>

  >>> tf.random.truncated_normal(shape=[2], mean=3, stddev=1, dtype=tf.float32)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([..., ...], dtype=float32)>

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the
      truncated normal distribution.
    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the normal distribution, before truncation.
    dtype: The type of the output. Restricted to floating-point types:
      `tf.half`, `tf.float`, `tf.double`, etc.
    seed: A Python integer. Used to create a random seed for the distribution.
      See `tf.random.set_seed` for more information.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random truncated normal values.
  """
  with ops.name_scope(name, "truncated_normal", [shape, mean, stddev]) as name:
    shape_tensor = shape_util.shape_tensor(shape)
    mean_tensor = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
    stddev_tensor = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
    seed1, seed2 = random_seed.get_seed(seed)
    rnd = gen_random_ops.truncated_normal(
        shape_tensor, dtype, seed=seed1, seed2=seed2)
    mul = rnd * stddev_tensor
    value = math_ops.add(mul, mean_tensor, name=name)
    shape_util.maybe_set_static_shape(value, shape)
    return value


ops.NotDifferentiable("ParameterizedTruncatedNormal")
ops.NotDifferentiable("TruncatedNormal")


@tf_export("random.uniform", v1=["random.uniform", "random_uniform"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("random_uniform")
def random_uniform(shape,
                   minval=0,
                   maxval=None,
                   dtype=dtypes.float32,
                   seed=None,
                   name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: an inline comment inside says an identity comparison is used for the fast path,
  # but the code actually uses an integer type check plus an equality test. Also, only Python
  # integers 0 and 1 trigger the fast path; float bounds like zero point zero go the slow path.
  # Creates random values uniformly distributed in the half-open range from minval (inclusive)
  # to maxval (exclusive). Inputs: shape (1-D integer tensor or sequence), minval (default 0),
  # maxval (default 1 for floating dtypes, mandatory for integer dtypes), dtype (float16,
  # bfloat16, float32, float64, int32 or int64; default float32), seed (optional integer) and
  # name (optional). Returns a tensor of the requested shape and dtype. Integer dtypes use the
  # integer uniform op directly with converted bounds; float dtypes sample in the unit range and
  # scale and shift as needed. Raises a value error for an unsupported dtype or when maxval is
  # missing with an integer dtype. Also sets the static shape of the result when known.
  """Outputs random values from a uniform distribution.

  The generated values follow a uniform distribution in the range
  `[minval, maxval)`. The lower bound `minval` is included in the range, while
  the upper bound `maxval` is excluded.

  For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
  be specified explicitly.

  In the integer case, the random integers are slightly biased unless
  `maxval - minval` is an exact power of two.  The bias is small for values of
  `maxval - minval` significantly smaller than the range of the output (either
  `2**32` or `2**64`).

  Examples:

  >>> tf.random.uniform(shape=[2])
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([..., ...], dtype=float32)>
  >>> tf.random.uniform(shape=[], minval=-1., maxval=0.)
  <tf.Tensor: shape=(), dtype=float32, numpy=-...>
  >>> tf.random.uniform(shape=[], minval=5, maxval=10, dtype=tf.int64)
  <tf.Tensor: shape=(), dtype=int64, numpy=...>

  The `seed` argument produces a deterministic sequence of tensors across
  multiple calls. To repeat that sequence, use `tf.random.set_seed`:

  >>> tf.random.set_seed(5)
  >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
  <tf.Tensor: shape=(), dtype=int32, numpy=2>
  >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
  <tf.Tensor: shape=(), dtype=int32, numpy=0>
  >>> tf.random.set_seed(5)
  >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
  <tf.Tensor: shape=(), dtype=int32, numpy=2>
  >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
  <tf.Tensor: shape=(), dtype=int32, numpy=0>

  Without `tf.random.set_seed` but with a `seed` argument is specified, small
  changes to function graphs or previously executed operations will change the
  returned value. See `tf.random.set_seed` for details.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    minval: A Tensor or Python value of type `dtype`, broadcastable with
      `shape` (for integer types, broadcasting is not supported, so it needs to
      be a scalar). The lower bound on the range of random values to generate
      (inclusive).  Defaults to 0.
    maxval: A Tensor or Python value of type `dtype`, broadcastable with
      `shape` (for integer types, broadcasting is not supported, so it needs to
      be a scalar). The upper bound on the range of random values to generate
      (exclusive). Defaults to 1 if `dtype` is floating point.
    dtype: The type of the output: `float16`, `bfloat16`, `float32`, `float64`,
      `int32`, or `int64`. Defaults to `float32`.
    seed: A Python integer. Used in combination with `tf.random.set_seed` to
      create a reproducible sequence of tensors across multiple calls.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random uniform values.

  Raises:
    ValueError: If `dtype` is integral and `maxval` is not specified.
  """
  dtype = dtypes.as_dtype(dtype)
  accepted_dtypes = (dtypes.float16, dtypes.bfloat16, dtypes.float32,
                     dtypes.float64, dtypes.int32, dtypes.int64)
  if dtype not in accepted_dtypes:
    raise ValueError(
        f"Argument `dtype` got invalid value {dtype}. Accepted dtypes are "
        f"{accepted_dtypes}.")
  if maxval is None:
    if dtype.is_integer:
      raise ValueError("Must specify maxval for integer dtype %r" % dtype)
    maxval = 1
  with ops.name_scope(name, "random_uniform", [shape, minval, maxval]) as name:
    shape = shape_util.shape_tensor(shape)
    # In case of [0,1) floating results, minval and maxval is unused. We do an
    # `is` comparison here since this is cheaper than isinstance or  __eq__.
    minval_is_zero = isinstance(minval, int) and minval == 0
    maxval_is_one = isinstance(maxval, int) and maxval == 1
    if not minval_is_zero or not maxval_is_one or dtype.is_integer:
      minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
      maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")
    seed1, seed2 = random_seed.get_seed(seed)
    if dtype.is_integer:
      result = gen_random_ops.random_uniform_int(
          shape, minval, maxval, seed=seed1, seed2=seed2, name=name)
    else:
      result = gen_random_ops.random_uniform(
          shape, dtype, seed=seed1, seed2=seed2)
      if minval_is_zero:
        if not maxval_is_one:
          result = math_ops.multiply(result, maxval)
      else:
        result = math_ops.add(result * (maxval - minval), minval, name=name)
    # TODO(b/132092188): C++ shape inference inside functional ops does not
    # cross FuncGraph boundaries since that information is only available in
    # python. So we manually get the static shape using
    # `constant_value_as_shape` which *does* cross function boundaries.
    shape_util.maybe_set_static_shape(result, shape)
    return result


ops.NotDifferentiable("RandomUniform")


@tf_export("random.shuffle", v1=["random.shuffle", "random_shuffle"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("random_shuffle")
def random_shuffle(value, seed=None, name=None):
  # [FUNCTION DOC]
  # Randomly permutes a tensor along its first dimension. Inputs: value (tensor to shuffle),
  # seed (optional integer combined with the global seed) and name (optional). Obtains the two
  # seeds and calls the raw shuffle op. Returns a tensor with the same shape and dtype as the
  # input. The input is not modified.
  """Randomly shuffles a tensor along its first dimension.

  The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
  to one and only one `output[i]`. For example, a mapping that might occur for a
  3x2 tensor is:

  ```python
  [[1, 2],       [[5, 6],
   [3, 4],  ==>   [1, 2],
   [5, 6]]        [3, 4]]
  ```

  Args:
    value: A Tensor to be shuffled.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      `tf.random.set_seed`
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of same shape and type as `value`, shuffled along its first
    dimension.
  """
  seed1, seed2 = random_seed.get_seed(seed)
  return gen_random_ops.random_shuffle(
      value, seed=seed1, seed2=seed2, name=name)


ops.NotDifferentiable("RandomShuffle")


@tf_export(v1=["random.multinomial", "multinomial"])
@dispatch.add_dispatch_support
@deprecation.deprecated(
    date=None, instructions="Use `tf.random.categorical` instead.")
def multinomial(logits, num_samples, seed=None, name=None, output_dtype=None):
  # [FUNCTION DOC]
  # Deprecated v1 entry point that draws class indices from a categorical distribution defined
  # by unnormalized log probabilities. Inputs: logits (2-D tensor of batch by classes),
  # num_samples (scalar count per row), seed (optional integer), name (optional) and output_dtype
  # (int32 or int64; None means int64). Opens a name scope and delegates to the shared
  # implementation. Returns an integer tensor of shape batch by num_samples. Invalid output
  # dtype raises a value error from the shared implementation. Superseded by the categorical
  # function.
  """Draws samples from a multinomial distribution.

  Example:

  ```python
  # samples has shape [1, 5], where each value is either 0 or 1 with equal
  # probability.
  samples = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 5)
  ```

  Args:
    logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice
      `[i, :]` represents the unnormalized log-probabilities for all classes.
    num_samples: 0-D.  Number of independent samples to draw for each row slice.
    seed: A Python integer. Used to create a random seed for the distribution.
      See `tf.random.set_seed` for behavior.
    name: Optional name for the operation.
    output_dtype: The integer type of the output: `int32` or `int64`. Defaults
      to `int64`.

  Returns:
    The drawn samples of shape `[batch_size, num_samples]`.
  """
  with ops.name_scope(name, "multinomial", [logits]):
    return multinomial_categorical_impl(logits, num_samples, output_dtype, seed)


@tf_export("random.categorical")
@dispatch.add_dispatch_support
def categorical(logits, num_samples, dtype=None, seed=None, name=None):
  # [FUNCTION DOC]
  # Draws samples from a categorical distribution. Inputs: logits (2-D tensor of batch by classes
  # holding unnormalized log probabilities), num_samples (scalar number of draws per row), dtype
  # (int32 or int64 output type; None means int64), seed (optional integer) and name (optional).
  # Opens a name scope and delegates to the shared implementation. Returns an integer tensor of
  # shape batch by num_samples. Raises a value error for an unsupported dtype.
  """Draws samples from a categorical distribution.

  Example:

  ```python
  # samples has shape [1, 5], where each value is either 0 or 1 with equal
  # probability.
  samples = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 5)
  ```

  Args:
    logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice
      `[i, :]` represents the unnormalized log-probabilities for all classes.
    num_samples: 0-D.  Number of independent samples to draw for each row slice.
    dtype: The integer type of the output: `int32` or `int64`. Defaults to
      `int64`.
    seed: A Python integer. Used to create a random seed for the distribution.
      See `tf.random.set_seed` for behavior.
    name: Optional name for the operation.

  Returns:
    The drawn samples of shape `[batch_size, num_samples]`.
  """
  with ops.name_scope(name, "categorical", [logits]):
    return multinomial_categorical_impl(logits, num_samples, dtype, seed)


def multinomial_categorical_impl(logits, num_samples, dtype, seed):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind the multinomial and categorical public functions.
  # Converts logits to a tensor, resolves dtype (missing means int64, only int32 and int64
  # accepted, otherwise raises a value error), derives the two seeds and calls the raw
  # multinomial op. Returns the integer sample tensor. Inputs: logits, num_samples, dtype, seed
  # (all required positional).
  """Implementation for random.categorical (v1) and random.categorical (v2)."""
  logits = ops.convert_to_tensor(logits, name="logits")
  dtype = dtypes.as_dtype(dtype) if dtype else dtypes.int64
  accepted_dtypes = (dtypes.int32, dtypes.int64)
  if dtype not in accepted_dtypes:
    raise ValueError(
        f"Argument `dtype` got invalid value {dtype}. Accepted dtypes are "
        f"{accepted_dtypes}.")
  seed1, seed2 = random_seed.get_seed(seed)
  return gen_random_ops.multinomial(
      logits, num_samples, seed=seed1, seed2=seed2, output_dtype=dtype)


ops.NotDifferentiable("Multinomial")


def _maybe_set_static_shape_helper(tensor, shape, postfix_tensor):
  # [FUNCTION DOC]
  # HELPER: shape-inference glue for the gamma and Poisson samplers.
  # Only when running in graph (non-eager) mode inside a function-building graph and the tensor
  # shape is not fully defined: computes the constant value of the sample shape, concatenates
  # the static shape of postfix_tensor to it and sets that as the tensor's static shape. This
  # recovers shape information that C++ inference loses across function boundaries. Otherwise
  # does nothing. Mutates the tensor's static shape; returns nothing.
  if (not context.executing_eagerly() and
      ops.get_default_graph().building_function and
      not tensor.shape.is_fully_defined()):
    shape = shape_util.shape_tensor(shape)
    const_shape = tensor_util.constant_value_as_shape(shape)
    postfix_tensor = ops.convert_to_tensor(postfix_tensor)
    tensor.set_shape(const_shape.concatenate(postfix_tensor.shape))


@tf_export("random.gamma", v1=["random.gamma", "random_gamma"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("random_gamma")
def random_gamma(shape,
                 alpha,
                 beta=None,
                 dtype=dtypes.float32,
                 seed=None,
                 name=None):
  # [FUNCTION DOC]
  # Draws samples from one or more Gamma distributions. Inputs: shape (1-D integer tensor of the
  # per-distribution sample shape), alpha (shape parameter, scalar or array), beta (inverse
  # scale, default 1, broadcast with alpha), dtype (float16, float32 or float64), seed
  # (optional) and name (optional). Alpha is broadcast against beta, samples are drawn with the
  # raw gamma op, divided by beta, and floored at the smallest positive value of the dtype to
  # avoid zero outcomes. Result shape is the sample shape followed by the broadcast parameter
  # shape. Sets the static shape via the shape helper. Samples are differentiable with respect
  # to alpha and beta.
  """Draws `shape` samples from each of the given Gamma distribution(s).

  `alpha` is the shape parameter describing the distribution(s), and `beta` is
  the inverse scale parameter(s).

  Note: Because internal calculations are done using `float64` and casting has
  `floor` semantics, we must manually map zero outcomes to the smallest
  possible positive floating-point value, i.e., `np.finfo(dtype).tiny`.  This
  means that `np.finfo(dtype).tiny` occurs more frequently than it otherwise
  should.  This bias can only happen for small values of `alpha`, i.e.,
  `alpha << 1` or large values of `beta`, i.e., `beta >> 1`.

  The samples are differentiable w.r.t. alpha and beta.
  The derivatives are computed using the approach described in
  (Figurnov et al., 2018).

  Example:

  ```python
  samples = tf.random.gamma([10], [0.5, 1.5])
  # samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents
  # the samples drawn from each distribution

  samples = tf.random.gamma([7, 5], [0.5, 1.5])
  # samples has shape [7, 5, 2], where each slice [:, :, 0] and [:, :, 1]
  # represents the 7x5 samples drawn from each of the two distributions

  alpha = tf.constant([[1.],[3.],[5.]])
  beta = tf.constant([[3., 4.]])
  samples = tf.random.gamma([30], alpha=alpha, beta=beta)
  # samples has shape [30, 3, 2], with 30 samples each of 3x2 distributions.

  loss = tf.reduce_mean(tf.square(samples))
  dloss_dalpha, dloss_dbeta = tf.gradients(loss, [alpha, beta])
  # unbiased stochastic derivatives of the loss function
  alpha.shape == dloss_dalpha.shape  # True
  beta.shape == dloss_dbeta.shape  # True
  ```

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output samples
      to be drawn per alpha/beta-parameterized distribution.
    alpha: A Tensor or Python value or N-D array of type `dtype`. `alpha`
      provides the shape parameter(s) describing the gamma distribution(s) to
      sample. Must be broadcastable with `beta`.
    beta: A Tensor or Python value or N-D array of type `dtype`. Defaults to 1.
      `beta` provides the inverse scale parameter(s) of the gamma
      distribution(s) to sample. Must be broadcastable with `alpha`.
    dtype: The type of alpha, beta, and the output: `float16`, `float32`, or
      `float64`.
    seed: A Python integer. Used to create a random seed for the distributions.
      See
      `tf.random.set_seed`
      for behavior.
    name: Optional name for the operation.

  Returns:
    samples: a `Tensor` of shape
      `tf.concat([shape, tf.shape(alpha + beta)], axis=0)` with values of type
      `dtype`.

  References:
    Implicit Reparameterization Gradients:
      [Figurnov et al., 2018]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
      ([pdf]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
  """
  with ops.name_scope(name, "random_gamma", [shape, alpha, beta]):
    shape = ops.convert_to_tensor(shape, name="shape", dtype=dtypes.int32)
    alpha = ops.convert_to_tensor(alpha, name="alpha", dtype=dtype)
    beta = ops.convert_to_tensor(
        beta if beta is not None else 1, name="beta", dtype=dtype)
    broadcast_shape = array_ops.broadcast_dynamic_shape(
        array_ops.shape(alpha), array_ops.shape(beta))
    alpha_broadcast = array_ops.broadcast_to(alpha, broadcast_shape)
    seed1, seed2 = random_seed.get_seed(seed)
    result = math_ops.maximum(
        np.finfo(alpha.dtype.as_numpy_dtype).tiny,
        gen_random_ops.random_gamma(
            shape, alpha_broadcast, seed=seed1, seed2=seed2) / beta)
    _maybe_set_static_shape_helper(result, shape, alpha_broadcast)
    return result


@tf_export(v1=["random.poisson", "random_poisson"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("random_poisson")
def random_poisson(lam, shape, dtype=dtypes.float32, seed=None, name=None):
  # [FUNCTION DOC]
  # Legacy (v1) Poisson sampler whose argument order is rate first, then sample shape. Inputs:
  # lam (rate parameter, scalar or array), shape (1-D integer sample shape), dtype (float16,
  # float32, float64, int32 or int64), seed (optional) and name (optional). Simply forwards to
  # the v2 function with the first two arguments swapped and returns its result.
  """Draws `shape` samples from each of the given Poisson distribution(s).

  `lam` is the rate parameter describing the distribution(s).

  Example:

  ```python
  samples = tf.random.poisson([0.5, 1.5], [10])
  # samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents
  # the samples drawn from each distribution

  samples = tf.random.poisson([12.2, 3.3], [7, 5])
  # samples has shape [7, 5, 2], where each slice [:, :, 0] and [:, :, 1]
  # represents the 7x5 samples drawn from each of the two distributions
  ```

  Args:
    lam: A Tensor or Python value or N-D array of type `dtype`.
      `lam` provides the rate parameter(s) describing the poisson
      distribution(s) to sample.
    shape: A 1-D integer Tensor or Python array. The shape of the output samples
      to be drawn per "rate"-parameterized distribution.
    dtype: The type of the output: `float16`, `float32`, `float64`, `int32` or
      `int64`.
    seed: A Python integer. Used to create a random seed for the distributions.
      See
      `tf.random.set_seed`
      for behavior.
    name: Optional name for the operation.

  Returns:
    samples: a `Tensor` of shape `tf.concat([shape, tf.shape(lam)], axis=0)`
      with values of type `dtype`.
  """
  return random_poisson_v2(shape, lam, dtype, seed, name)


@tf_export("random.poisson", v1=[])
@dispatch.add_dispatch_support
def random_poisson_v2(shape, lam, dtype=dtypes.float32, seed=None, name=None):
  # [FUNCTION DOC]
  # Draws samples from Poisson distributions. Inputs: shape (1-D integer sample shape, converted
  # to an int32 tensor), lam (rate parameter, scalar or array), dtype (float16, float32,
  # float64, int32 or int64; default float32), seed (optional) and name (optional). Calls the raw
  # Poisson op with both seeds, then sets the static shape through the shape helper. Returns a
  # tensor of shape sample shape followed by the shape of lam.
  """Draws `shape` samples from each of the given Poisson distribution(s).

  `lam` is the rate parameter describing the distribution(s).

  Example:

  ```python
  samples = tf.random.poisson([10], [0.5, 1.5])
  # samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents
  # the samples drawn from each distribution

  samples = tf.random.poisson([7, 5], [12.2, 3.3])
  # samples has shape [7, 5, 2], where each slice [:, :, 0] and [:, :, 1]
  # represents the 7x5 samples drawn from each of the two distributions
  ```

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output samples
      to be drawn per "rate"-parameterized distribution.
    lam: A Tensor or Python value or N-D array of type `dtype`.
      `lam` provides the rate parameter(s) describing the poisson
      distribution(s) to sample.
    dtype: The type of the output: `float16`, `float32`, `float64`, `int32` or
      `int64`.
    seed: A Python integer. Used to create a random seed for the distributions.
      See
      `tf.random.set_seed`
      for behavior.
    name: Optional name for the operation.

  Returns:
    samples: a `Tensor` of shape `tf.concat([shape, tf.shape(lam)], axis=0)`
      with values of type `dtype`.
  """
  with ops.name_scope(name, "random_poisson", [lam, shape]):
    shape = ops.convert_to_tensor(shape, name="shape", dtype=dtypes.int32)
    seed1, seed2 = random_seed.get_seed(seed)
    result = gen_random_ops.random_poisson_v2(
        shape, lam, dtype=dtype, seed=seed1, seed2=seed2)
    _maybe_set_static_shape_helper(result, shape, lam)
    return result
