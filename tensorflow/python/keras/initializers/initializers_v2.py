# [MODULE DOC]
# Keras initializers for TF2 behavior. Defines the Initializer base class and the concrete
# initializers (zeros, ones, constant, random uniform, random normal, truncated normal, variance
# scaling, orthogonal, identity, and the Glorot, Lecun and He variants), plus private helpers for
# dtype validation, fan computation, keyword validation and seeded random generation.
# Initializers are callable objects taking a shape and optional dtype and returning a tensor; they
# also support configuration round trips. Used by the initializers package registry when TF2
# behavior is enabled. No import-time side effects beyond defining names and two private
# constants naming the partition keyword arguments.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Keras initializers for TF 2."""
# pylint: disable=g-classes-have-attributes

import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops

_PARTITION_SHAPE = 'partition_shape'
_PARTITION_OFFSET = 'partition_offset'


class Initializer(object):
  # [CLASS DOC]
  # Abstract base class of all Keras version 2 initializers. Subclasses implement calling with a
  # shape and dtype to produce a tensor, and optionally a configuration method for serialization.
  # Holds no state itself.
  """Initializer base class: all Keras initializers inherit from this class.

  Initializers should implement a `__call__` method with the following
  signature:

  ```python
  def __call__(self, shape, dtype=None, **kwargs):
    # returns a tensor of shape `shape` and dtype `dtype`
    # containing values drawn from a distribution of your choice.
  ```

  Optionally, you an also implement the method `get_config` and the class
  method `from_config` in order to support serialization -- just like with
  any Keras object.

  Here's a simple example: a random normal initializer.

  ```python
  import tensorflow as tf

  class ExampleRandomNormal(tf.keras.initializers.Initializer):

    def __init__(self, mean, stddev):
      self.mean = mean
      self.stddev = stddev

    def __call__(self, shape, dtype=None, **kwargs):
      return tf.random.normal(
          shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

    def get_config(self):  # To support serialization
      return {"mean": self.mean, "stddev": self.stddev}
  ```

  Note that we don't have to implement `from_config` in the example above since
  the constructor arguments of the class the keys in the config returned by
  `get_config` are the same. In this case, the default `from_config`
  works fine.
  """

  def __call__(self, shape, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Abstract method. Inputs: tensor shape, optional dtype, extra keyword arguments. Always
    # raises a not-implemented error; subclasses must override it to return an initialized tensor.
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor.
      **kwargs: Additional keyword arguments.
    """
    raise NotImplementedError

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the serializable configuration of the initializer. The base version returns an
    # empty mapping; subclasses with constructor arguments override it.
    """Returns the configuration of the initializer as a JSON-serializable dict.

    Returns:
      A JSON-serializable Python dict.
    """
    return {}

  @classmethod
  def from_config(cls, config):
    # [FUNCTION DOC]
    # Class method that creates an instance from a configuration mapping (the output of the
    # configuration method). Input: mapping of constructor argument names to values. It removes
    # any dtype entry from the mapping (mutating the caller's mapping) and passes the remaining
    # entries to the constructor. Returns the new initializer.
    """Instantiates an initializer from a configuration dictionary.

    Example:

    ```python
    initializer = RandomUniform(-1, 1)
    config = initializer.get_config()
    initializer = RandomUniform.from_config(config)
    ```

    Args:
      config: A Python dictionary, the output of `get_config`.

    Returns:
      A `tf.keras.initializers.Initializer` instance.
    """
    config.pop('dtype', None)
    return cls(**config)


class Zeros(Initializer):
  # [CLASS DOC]
  # Initializer that produces tensors filled with zeros. Stateless.
  """Initializer that generates tensors initialized to 0.

  Also available via the shortcut function `tf.keras.initializers.zeros`.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Zeros()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Zeros()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
  """

  def __call__(self, shape, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Returns a tensor of zeros. Inputs: shape; optional dtype (default is the Keras backend float
    # type; must be numeric or boolean); keyword arguments, of which only the partition shape and
    # partition offset are accepted. If a partition shape is given it replaces the shape.
    # Raises a type error for unknown keywords and a value error for a string or non-numeric
    # dtype.
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are
       supported. If not specified, `tf.keras.backend.floatx()` is used,
       which default to `float32` unless you configured it otherwise
       (via `tf.keras.backend.set_floatx(float_dtype)`).
      **kwargs: Additional keyword arguments.
    """
    _validate_kwargs(self.__class__.__name__, kwargs)
    dtype = _get_dtype(dtype)
    if not dtype.is_numpy_compatible or dtype == dtypes.string:
      raise ValueError('Expected numeric or boolean dtype, got %s.' % dtype)
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    return array_ops.zeros(shape, dtype)


class Ones(Initializer):
  # [CLASS DOC]
  # Initializer that produces tensors filled with ones. Stateless.
  """Initializer that generates tensors initialized to 1.

  Also available via the shortcut function `tf.keras.initializers.ones`.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Ones()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Ones()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
  """

  def __call__(self, shape, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Returns a tensor of ones. Inputs: shape; optional dtype (default is the Keras backend float
    # type; must be numeric or boolean); keyword arguments, of which only the partition shape and
    # partition offset are accepted. If a partition shape is given it replaces the shape.
    # Raises a type error for unknown keywords and a value error for a non-numeric dtype.
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are
       supported. If not specified, `tf.keras.backend.floatx()` is used,
       which default to `float32` unless you configured it otherwise
       (via `tf.keras.backend.set_floatx(float_dtype)`).
      **kwargs: Additional keyword arguments.
    """
    _validate_kwargs(self.__class__.__name__, kwargs)
    dtype = _get_dtype(dtype)
    if not dtype.is_numpy_compatible or dtype == dtypes.string:
      raise ValueError('Expected numeric or boolean dtype, got %s.' % dtype)
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    return array_ops.ones(shape, dtype)


class Constant(Initializer):
  # [CLASS DOC]
  # Initializer that fills tensors with a single constant value, stored as an attribute.
  """Initializer that generates tensors with constant values.

  Also available via the shortcut function `tf.keras.initializers.constant`.

  Only scalar values are allowed.
  The constant value provided must be convertible to the dtype requested
  when calling the initializer.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Constant(3.)
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Constant(3.)
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    value: A Python scalar.
  """

  def __init__(self, value=0):
    # [FUNCTION DOC]
    # Constructor. Input: the value to fill with (default 0). Stores it.
    self.value = value

  def __call__(self, shape, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Returns a constant tensor of the given shape filled with the stored value, converted to the
    # dtype (default is the Keras backend float type). Extra keyword arguments are ignored,
    # including partition arguments, which are not applied.
    """Returns a tensor object initialized to `self.value`.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. If not specified,
       `tf.keras.backend.floatx()` is used,
       which default to `float32` unless you configured it otherwise
       (via `tf.keras.backend.set_floatx(float_dtype)`).
      **kwargs: Additional keyword arguments.
    """
    del kwargs
    return constant_op.constant(
        self.value, dtype=_get_dtype(dtype), shape=shape)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the constant value.
    return {'value': self.value}


class RandomUniform(Initializer):
  # [CLASS DOC]
  # Initializer drawing values from a uniform distribution between a lower and upper bound,
  # using a seeded stateless generator when a seed is given. Holds minval, maxval and seed.
  """Initializer that generates tensors with a uniform distribution.

  Also available via the shortcut function
  `tf.keras.initializers.random_uniform`.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range of
      random values to generate (inclusive).
    maxval: A python scalar or a scalar tensor. Upper bound of the range of
      random values to generate (exclusive).
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.
  """

  def __init__(self, minval=-0.05, maxval=0.05, seed=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: lower bound (default minus 0.05), upper bound (default 0.05), optional
    # seed (default none, meaning nondeterministic). Stores them and creates a random generator
    # helper for the seed.
    self.minval = minval
    self.maxval = maxval
    self.seed = seed
    self._random_generator = _RandomGenerator(seed)

  def __call__(self, shape, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Returns a tensor of uniform random values. Inputs: shape; optional dtype (default is the
    # Keras backend float type; float or integer only); keywords limited to partition shape and
    # offset (partition shape replaces the shape). Raises a type error for unknown keywords and a
    # value error for other dtypes. Deterministic for a given seed.
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point and integer
      types are supported. If not specified,
        `tf.keras.backend.floatx()` is used,
       which default to `float32` unless you configured it otherwise
       (via `tf.keras.backend.set_floatx(float_dtype)`).
      **kwargs: Additional keyword arguments.
    """
    _validate_kwargs(self.__class__.__name__, kwargs)
    dtype = _get_dtype(dtype)
    if not dtype.is_floating and not dtype.is_integer:
      raise ValueError('Expected float or integer dtype, got %s.' % dtype)
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    return self._random_generator.random_uniform(shape, self.minval,
                                                 self.maxval, dtype)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the lower bound, upper bound and seed.
    return {
        'minval': self.minval,
        'maxval': self.maxval,
        'seed': self.seed
    }


class RandomNormal(Initializer):
  # [CLASS DOC]
  # Initializer drawing values from a normal distribution with a given mean and standard
  # deviation, using a seeded stateless generator when a seed is given.
  """Initializer that generates tensors with a normal distribution.

  Also available via the shortcut function
  `tf.keras.initializers.random_normal`.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values to
      generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the random
      values to generate.
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.
  """

  def __init__(self, mean=0.0, stddev=0.05, seed=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: mean (default 0.0), standard deviation (default 0.05), optional seed.
    # Stores them and creates a random generator helper.
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    self._random_generator = _RandomGenerator(seed)

  def __call__(self, shape, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Returns a tensor of normally distributed values. Inputs: shape; optional dtype (default is
    # the Keras backend float type; must be floating point); keywords limited to partition shape
    # and offset (partition shape replaces the shape). Raises a type error for unknown keywords
    # and a value error for non-floating dtypes.
    """Returns a tensor object initialized to random normal values.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
        supported. If not specified, `tf.keras.backend.floatx()` is used, which
        default to `float32` unless you configured it otherwise (via
        `tf.keras.backend.set_floatx(float_dtype)`)
      **kwargs: Additional keyword arguments.
    """
    _validate_kwargs(self.__class__.__name__, kwargs)
    dtype = _assert_float_dtype(_get_dtype(dtype))
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    return self._random_generator.random_normal(shape, self.mean, self.stddev,
                                                dtype)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the mean, standard deviation and seed.
    return {
        'mean': self.mean,
        'stddev': self.stddev,
        'seed': self.seed
    }


class TruncatedNormal(Initializer):
  # [CLASS DOC]
  # Initializer drawing from a normal distribution truncated at two standard deviations from the
  # mean, so out-of-range values are redrawn. Holds mean, standard deviation and seed.
  """Initializer that generates a truncated normal distribution.

  Also available via the shortcut function
  `tf.keras.initializers.truncated_normal`.

  The values generated are similar to values from a
  `tf.keras.initializers.RandomNormal` initializer except that values more
  than two standard deviations from the mean are
  discarded and re-drawn.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate before truncation.
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.
  """

  def __init__(self, mean=0.0, stddev=0.05, seed=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: mean (default 0.0), standard deviation (default 0.05), optional seed.
    # Stores them and creates a random generator helper.
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    self._random_generator = _RandomGenerator(seed)

  def __call__(self, shape, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Returns a tensor of truncated normal values. Inputs: shape; optional dtype (default is the
    # Keras backend float type; must be floating point); keywords limited to partition shape and
    # offset (partition shape replaces the shape). Raises a type error for unknown keywords and a
    # value error for non-floating dtypes.
    """Returns a tensor object initialized to random normal values (truncated).

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
        supported. If not specified, `tf.keras.backend.floatx()` is used, which
        default to `float32` unless you configured it otherwise (via
        `tf.keras.backend.set_floatx(float_dtype)`)
      **kwargs: Additional keyword arguments.
    """
    _validate_kwargs(self.__class__.__name__, kwargs)
    dtype = _assert_float_dtype(_get_dtype(dtype))
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    return self._random_generator.truncated_normal(shape, self.mean,
                                                   self.stddev, dtype)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the mean, standard deviation and seed.
    return {
        'mean': self.mean,
        'stddev': self.stddev,
        'seed': self.seed
    }


class VarianceScaling(Initializer):
  # [CLASS DOC]
  # Initializer whose spread adapts to the tensor shape. The scale is divided by the number of
  # input units, output units or their average (the mode), then values are drawn from a truncated
  # normal, untruncated normal or uniform distribution with matching variance. Holds scale, mode,
  # distribution and seed. Base class of the Glorot, Lecun and He initializers.
  """Initializer capable of adapting its scale to the shape of weights tensors.

  Also available via the shortcut function
  `tf.keras.initializers.variance_scaling`.

  With `distribution="truncated_normal" or "untruncated_normal"`, samples are
  drawn from a truncated/untruncated normal distribution with a mean of zero and
  a standard deviation (after truncation, if used) `stddev = sqrt(scale / n)`,
  where `n` is:

  - number of input units in the weight tensor, if `mode="fan_in"`
  - number of output units, if `mode="fan_out"`
  - average of the numbers of input and output units, if `mode="fan_avg"`

  With `distribution="uniform"`, samples are drawn from a uniform distribution
  within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.VarianceScaling(
  ... scale=0.1, mode='fan_in', distribution='uniform')
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.VarianceScaling(
  ... scale=0.1, mode='fan_in', distribution='uniform')
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    scale: Scaling factor (positive float).
    mode: One of "fan_in", "fan_out", "fan_avg".
    distribution: Random distribution to use. One of "truncated_normal",
      "untruncated_normal" and  "uniform".
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.
  """

  def __init__(self,
               scale=1.0,
               mode='fan_in',
               distribution='truncated_normal',
               seed=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: scale (positive number, default 1.0), mode (one of fan in, fan out or
    # fan average; default fan in), distribution (uniform, truncated normal or untruncated
    # normal, case-insensitive; the name normal is treated as truncated normal; default truncated
    # normal), optional seed. Raises a value error for a non-positive scale, invalid mode or
    # invalid distribution. Stores the settings and creates a random generator helper.
    if scale <= 0.:
      raise ValueError('`scale` must be positive float.')
    if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
      raise ValueError('Invalid `mode` argument:', mode)
    distribution = distribution.lower()
    # Compatibility with keras-team/keras.
    if distribution == 'normal':
      distribution = 'truncated_normal'
    if distribution not in {'uniform', 'truncated_normal',
                            'untruncated_normal'}:
      raise ValueError('Invalid `distribution` argument:', distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed
    self._random_generator = _RandomGenerator(seed)

  def __call__(self, shape, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Returns a tensor with variance scaled by the shape. Inputs: shape; optional dtype (default
    # is the Keras backend float type; must be floating point); keywords limited to partition
    # shape and offset. Fan values are computed from the full shape before a partition shape
    # replaces it. Standard deviation is the square root of the adjusted scale (divided by a
    # constant for the truncated case); the uniform limit is the square root of three times the
    # adjusted scale. Raises a type error for unknown keywords and a value error for
    # non-floating dtypes.
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
        supported. If not specified, `tf.keras.backend.floatx()` is used, which
        default to `float32` unless you configured it otherwise (via
        `tf.keras.backend.set_floatx(float_dtype)`)
      **kwargs: Additional keyword arguments.
    """
    _validate_kwargs(self.__class__.__name__, kwargs)
    dtype = _assert_float_dtype(_get_dtype(dtype))
    scale = self.scale
    fan_in, fan_out = _compute_fans(shape)
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    if self.mode == 'fan_in':
      scale /= max(1., fan_in)
    elif self.mode == 'fan_out':
      scale /= max(1., fan_out)
    else:
      scale /= max(1., (fan_in + fan_out) / 2.)
    if self.distribution == 'truncated_normal':
      # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      stddev = math.sqrt(scale) / .87962566103423978
      return self._random_generator.truncated_normal(shape, 0.0, stddev, dtype)
    elif self.distribution == 'untruncated_normal':
      stddev = math.sqrt(scale)
      return self._random_generator.random_normal(shape, 0.0, stddev, dtype)
    else:
      limit = math.sqrt(3.0 * scale)
      return self._random_generator.random_uniform(shape, -limit, limit, dtype)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with scale, mode, distribution and seed.
    return {
        'scale': self.scale,
        'mode': self.mode,
        'distribution': self.distribution,
        'seed': self.seed
    }


class Orthogonal(Initializer):
  # [CLASS DOC]
  # Initializer producing a (semi-)orthogonal matrix via QR factorization of random normal values,
  # multiplied by a gain. Holds gain and seed. Partitioned initialization is not supported.
  """Initializer that generates an orthogonal matrix.

  Also available via the shortcut function `tf.keras.initializers.orthogonal`.

  If the shape of the tensor to initialize is two-dimensional, it is initialized
  with an orthogonal matrix obtained from the QR decomposition of a matrix of
  random numbers drawn from a normal distribution.
  If the matrix has fewer rows than columns then the output will have orthogonal
  rows. Otherwise, the output will have orthogonal columns.

  If the shape of the tensor to initialize is more than two-dimensional,
  a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
  is initialized, where `n` is the length of the shape vector.
  The matrix is subsequently reshaped to give a tensor of the desired shape.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Orthogonal()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Orthogonal()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    gain: multiplicative factor to apply to the orthogonal matrix
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

  References:
      [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
      ([pdf](https://arxiv.org/pdf/1312.6120.pdf))
  """

  def __init__(self, gain=1.0, seed=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: multiplicative gain (default 1.0), optional seed. Stores them and
    # creates a random generator helper.
    self.gain = gain
    self.seed = seed
    self._random_generator = _RandomGenerator(seed)

  def __call__(self, shape, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Returns an orthogonal tensor. Inputs: shape (at least two dimensions; all but the last are
    # flattened into rows), optional dtype (floating point, default Keras backend float type),
    # and keywords, none of which are allowed in practice. Generates a random matrix, takes the
    # QR factorization, fixes signs using the diagonal of the triangular factor, transposes if
    # rows are fewer than columns, scales by the gain and reshapes. Raises a type error for
    # unknown keywords, a value error for partition keywords, a non-floating dtype or fewer
    # than two dimensions.
    """Returns a tensor object initialized to an orthogonal matrix.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
        supported. If not specified, `tf.keras.backend.floatx()` is used,
       which default to `float32` unless you configured it otherwise
       (via `tf.keras.backend.set_floatx(float_dtype)`)
      **kwargs: Additional keyword arguments.
    """
    _validate_kwargs(self.__class__.__name__, kwargs, support_partition=False)
    dtype = _assert_float_dtype(_get_dtype(dtype))
    # Check the shape
    if len(shape) < 2:
      raise ValueError('The tensor to initialize must be '
                       'at least two-dimensional')
    # Flatten the input shape with the last dimension remaining
    # its original shape so it works for conv2d
    num_rows = 1
    for dim in shape[:-1]:
      num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))

    # Generate a random matrix
    a = self._random_generator.random_normal(flat_shape, dtype=dtype)
    # Compute the qr factorization
    q, r = gen_linalg_ops.qr(a, full_matrices=False)
    # Make Q uniform
    d = array_ops.tensor_diag_part(r)
    q *= math_ops.sign(d)
    if num_rows < num_cols:
      q = array_ops.matrix_transpose(q)
    return self.gain * array_ops.reshape(q, shape)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the gain and seed.
    return {'gain': self.gain, 'seed': self.seed}


class Identity(Initializer):
  # [CLASS DOC]
  # Initializer producing a two-dimensional identity matrix scaled by a gain. Holds the gain.
  # Partitioned initialization is not supported.
  """Initializer that generates the identity matrix.

  Also available via the shortcut function `tf.keras.initializers.identity`.

  Only usable for generating 2D matrices.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Identity()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Identity()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    gain: Multiplicative factor to apply to the identity matrix.
  """

  def __init__(self, gain=1.0):
    # [FUNCTION DOC]
    # Constructor. Input: multiplicative gain (default 1.0). Stores it.
    self.gain = gain

  def __call__(self, shape, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Returns a scaled identity matrix. Inputs: shape (must have exactly two dimensions),
    # optional dtype (floating point, default Keras backend float type), keywords (partition
    # keywords are rejected). Raises a type error for unknown keywords, a value error for partition
    # keywords, a non-floating dtype or a shape that is not two-dimensional.
    """Returns a tensor object initialized to a 2D identity matrix.

    Args:
      shape: Shape of the tensor. It should have exactly rank 2.
      dtype: Optional dtype of the tensor. Only floating point types are
       supported. If not specified, `tf.keras.backend.floatx()` is used,
       which default to `float32` unless you configured it otherwise
       (via `tf.keras.backend.set_floatx(float_dtype)`)
      **kwargs: Additional keyword arguments.
    """
    _validate_kwargs(self.__class__.__name__, kwargs, support_partition=False)
    dtype = _assert_float_dtype(_get_dtype(dtype))
    if len(shape) != 2:
      raise ValueError(
          'Identity matrix initializer can only be used for 2D matrices.')
    initializer = linalg_ops.eye(*shape, dtype=dtype)
    return self.gain * initializer

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the gain.
    return {'gain': self.gain}


class GlorotUniform(VarianceScaling):
  # [CLASS DOC]
  # Glorot (Xavier) uniform initializer: variance scaling with scale one, fan average mode and
  # uniform distribution. Only the seed is configurable.
  """The Glorot uniform initializer, also called Xavier uniform initializer.

  Also available via the shortcut function
  `tf.keras.initializers.glorot_uniform`.

  Draws samples from a uniform distribution within `[-limit, limit]`, where
  `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input units
  in the weight tensor and `fan_out` is the number of output units).

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.GlorotUniform()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.GlorotUniform()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

  References:
      [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
      ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """

  def __init__(self, seed=None):
    # [FUNCTION DOC]
    # Constructor. Input: optional random seed. Configures the parent variance-scaling
    # initializer with this class's fixed scale, mode and distribution.
    super(GlorotUniform, self).__init__(
        scale=1.0,
        mode='fan_avg',
        distribution='uniform',
        seed=seed)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping containing only the seed; the fixed scale, mode and distribution are not
    # serialized.
    return {'seed': self.seed}


class GlorotNormal(VarianceScaling):
  # [CLASS DOC]
  # Glorot (Xavier) normal initializer: variance scaling with scale one, fan average mode and
  # truncated normal distribution. Only the seed is configurable.
  """The Glorot normal initializer, also called Xavier normal initializer.

  Also available via the shortcut function
  `tf.keras.initializers.glorot_normal`.

  Draws samples from a truncated normal distribution centered on 0 with `stddev
  = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of input units in
  the weight tensor and `fan_out` is the number of output units in the weight
  tensor.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.GlorotNormal()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.GlorotNormal()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

  References:
      [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
      ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """

  def __init__(self, seed=None):
    # [FUNCTION DOC]
    # Constructor. Input: optional random seed. Configures the parent variance-scaling
    # initializer with this class's fixed scale, mode and distribution.
    super(GlorotNormal, self).__init__(
        scale=1.0,
        mode='fan_avg',
        distribution='truncated_normal',
        seed=seed)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping containing only the seed; the fixed scale, mode and distribution are not
    # serialized.
    return {'seed': self.seed}


class LecunNormal(VarianceScaling):
  # [CLASS DOC]
  # Lecun normal initializer: variance scaling with scale one, fan in mode and truncated normal
  # distribution. Only the seed is configurable.
  """Lecun normal initializer.

   Also available via the shortcut function
  `tf.keras.initializers.lecun_normal`.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Draws samples from a truncated normal distribution centered on 0 with `stddev
  = sqrt(1 / fan_in)` where `fan_in` is the number of input units in the weight
  tensor.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.LecunNormal()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.LecunNormal()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    seed: A Python integer. Used to seed the random generator.

  References:
      - Self-Normalizing Neural Networks,
      [Klambauer et al., 2017]
      (https://papers.nips.cc/paper/6698-self-normalizing-neural-networks)
      ([pdf]
      (https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf))
      - Efficient Backprop,
      [Lecun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  """

  def __init__(self, seed=None):
    # [FUNCTION DOC]
    # Constructor. Input: optional random seed. Configures the parent variance-scaling
    # initializer with this class's fixed scale, mode and distribution.
    super(LecunNormal, self).__init__(
        scale=1., mode='fan_in', distribution='truncated_normal', seed=seed)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping containing only the seed; the fixed scale, mode and distribution are not
    # serialized.
    return {'seed': self.seed}


class LecunUniform(VarianceScaling):
  # [CLASS DOC]
  # Lecun uniform initializer: variance scaling with scale one, fan in mode and uniform
  # distribution. Only the seed is configurable.
  """Lecun uniform initializer.

   Also available via the shortcut function
  `tf.keras.initializers.lecun_uniform`.

  Draws samples from a uniform distribution within `[-limit, limit]`,
  where `limit = sqrt(3 / fan_in)` (`fan_in` is the number of input units in the
  weight tensor).

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.LecunUniform()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.LecunUniform()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

  References:
      - Self-Normalizing Neural Networks,
      [Klambauer et al., 2017](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks) # pylint: disable=line-too-long
      ([pdf](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf))
      - Efficient Backprop,
      [Lecun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  """

  def __init__(self, seed=None):
    # [FUNCTION DOC]
    # Constructor. Input: optional random seed. Configures the parent variance-scaling
    # initializer with this class's fixed scale, mode and distribution.
    super(LecunUniform, self).__init__(
        scale=1., mode='fan_in', distribution='uniform', seed=seed)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping containing only the seed; the fixed scale, mode and distribution are not
    # serialized.
    return {'seed': self.seed}


class HeNormal(VarianceScaling):
  # [CLASS DOC]
  # He normal initializer: variance scaling with scale two, fan in mode and truncated normal
  # distribution. Only the seed is configurable.
  """He normal initializer.

   Also available via the shortcut function
  `tf.keras.initializers.he_normal`.

  It draws samples from a truncated normal distribution centered on 0 with
  `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in the
  weight tensor.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.HeNormal()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.HeNormal()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

  References:
      [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) # pylint: disable=line-too-long
      ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
  """

  def __init__(self, seed=None):
    # [FUNCTION DOC]
    # Constructor. Input: optional random seed. Configures the parent variance-scaling
    # initializer with this class's fixed scale, mode and distribution.
    super(HeNormal, self).__init__(
        scale=2., mode='fan_in', distribution='truncated_normal', seed=seed)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping containing only the seed; the fixed scale, mode and distribution are not
    # serialized.
    return {'seed': self.seed}


class HeUniform(VarianceScaling):
  # [CLASS DOC]
  # He uniform initializer: variance scaling with scale two, fan in mode and uniform
  # distribution. Only the seed is configurable.
  """He uniform variance scaling initializer.

   Also available via the shortcut function
  `tf.keras.initializers.he_uniform`.

  Draws samples from a uniform distribution within `[-limit, limit]`, where
  `limit = sqrt(6 / fan_in)` (`fan_in` is the number of input units in the
  weight tensor).

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.HeUniform()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.HeUniform()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

  References:
      [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) # pylint: disable=line-too-long
      ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
  """

  def __init__(self, seed=None):
    # [FUNCTION DOC]
    # Constructor. Input: optional random seed. Configures the parent variance-scaling
    # initializer with this class's fixed scale, mode and distribution.
    super(HeUniform, self).__init__(
        scale=2., mode='fan_in', distribution='uniform', seed=seed)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping containing only the seed; the fixed scale, mode and distribution are not
    # serialized.
    return {'seed': self.seed}


def _get_dtype(dtype):
  # [FUNCTION DOC]
  # HELPER: dtype normalization for the initializers in this module.
  # Converts a dtype (name, numpy type or TF type) to a TF dtype; when none is given uses the
  # Keras backend default float type. Returns the TF dtype. Invalid types raise from the
  # conversion.
  if dtype is None:
    dtype = backend.floatx()
  return dtypes.as_dtype(dtype)


def _assert_float_dtype(dtype):
  # [FUNCTION DOC]
  # HELPER: dtype validation for floating-point initializers.
  # Converts the input to a TF dtype and returns it if it is floating point; otherwise raises a
  # value error.
  """Validate and return floating point type based on `dtype`.

  `dtype` must be a floating point type.

  Args:
    dtype: The data type to validate.

  Returns:
    Validated type.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
  dtype = dtypes.as_dtype(dtype)
  if not dtype.is_floating:
    raise ValueError('Expected floating point type, got %s.' % dtype)
  return dtype


class _RandomGenerator(object):
  # [CLASS DOC]
  # HELPER: seeded random value source for the initializers in this module.
  # Holds an optional seed, stored as a two-integer pair as required by stateless random ops.
  # With a seed it uses stateless (deterministic) random operations, otherwise ordinary stateful
  # ones.
  """Random generator that selects appropriate random ops."""

  def __init__(self, seed=None):
    # [FUNCTION DOC]
    # Constructor. Input: optional integer seed. If given, stores it as a two-element pair (the
    # seed and zero) as needed by stateless random ops; otherwise stores nothing (null value).
    super(_RandomGenerator, self).__init__()
    if seed is not None:
      # Stateless random ops requires 2-int seed.
      self.seed = [seed, 0]
    else:
      self.seed = None

  def random_normal(self, shape, mean=0.0, stddev=1, dtype=dtypes.float32):
    # [FUNCTION DOC]
    # Returns normally distributed random values. Inputs: shape, mean (default 0), standard
    # deviation (default 1), dtype (default float32). Uses the stateless op when a seed is stored
    # (deterministic) and the ordinary op otherwise.
    """A deterministic random normal if seed is passed."""
    if self.seed:
      op = stateless_random_ops.stateless_random_normal
    else:
      op = random_ops.random_normal
    return op(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.seed)

  def random_uniform(self, shape, minval, maxval, dtype):
    # [FUNCTION DOC]
    # Returns uniformly distributed random values. Inputs: shape, lower bound, upper bound, dtype.
    # Uses the stateless op when a seed is stored (deterministic) and the ordinary op otherwise.
    """A deterministic random uniform if seed is passed."""
    if self.seed:
      op = stateless_random_ops.stateless_random_uniform
    else:
      op = random_ops.random_uniform
    return op(
        shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=self.seed)

  def truncated_normal(self, shape, mean, stddev, dtype):
    # [FUNCTION DOC]
    # Returns truncated normal random values. Inputs: shape, mean, standard deviation, dtype.
    # Uses the stateless op when a seed is stored (deterministic) and the ordinary op otherwise.
    """A deterministic truncated normal if seed is passed."""
    if self.seed:
      op = stateless_random_ops.stateless_truncated_normal
    else:
      op = random_ops.truncated_normal
    return op(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.seed)


def _compute_fans(shape):
  # [FUNCTION DOC]
  # HELPER: shape analysis for variance-scaling initializers.
  # Computes the number of input and output units of a weight shape. An empty shape gives one and
  # one; a one-dimensional shape gives its length for both; two dimensions give first and second
  # sizes; higher ranks are treated as convolution kernels, multiplying the last two sizes by the
  # product of the leading sizes. Returns a pair of integers (fan in, fan out).
  """Computes the number of input and output units for a weight shape.

  Args:
    shape: Integer shape tuple or TF tensor shape.

  Returns:
    A tuple of integer scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return int(fan_in), int(fan_out)


def _validate_kwargs(cls_name, kwargs, support_partition=True):
  # [FUNCTION DOC]
  # HELPER: argument checking for initializer calls.
  # Inputs: class name (for messages), mapping of extra keyword arguments, and whether partition
  # arguments are allowed (default yes). Raises a type error for any keyword other than the
  # partition shape and offset, and a value error naming the class if a partition keyword is
  # given while partitioning is unsupported. Returns nothing.
  for kwarg in kwargs:
    if kwarg not in [_PARTITION_SHAPE, _PARTITION_OFFSET]:
      raise TypeError('Unknown keyword arguments: %s' % kwarg)
    elif not support_partition:
      raise ValueError('%s initializer doesn\'t support partition-related '
                       'arguments' % cls_name)
