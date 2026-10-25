# [MODULE DOC]
# Random number functions of the experimental NumPy-compatible API (random.seed, randn,
# standard_normal, uniform, poisson, random, rand, randint). Each function is a thin wrapper over
# TensorFlow random ops (random_normal, random_uniform, random_poisson) and is exported under the
# experimental numpy random namespace through the export decorator and wrapped with the numpy
# documentation decorator. Results are tensors/ndarrays whose float dtype follows the numpy
# dtype configuration helpers. Exposes the module constant DEFAULT_RANDN_DTYPE (single precision
# float, marked for deprecation). Import-time side effects: registers the exported symbols only.
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
"""Random functions."""

# pylint: disable=g-direct-tensorflow-import

import numpy as onp

from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export

# TODO(agarwal): deprecate this.
DEFAULT_RANDN_DTYPE = onp.float32


@tf_export.tf_export('experimental.numpy.random.seed', v1=[])
@np_utils.np_doc('random.seed')
def seed(s):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it uses the old set_random_seed; it actually calls the
  # framework random_seed set_seed function. Also, only type errors from the integer conversion
  # are re-raised with the friendly message; a non-numeric string raises the raw conversion error.
  # Sets the global random seed. Input s: anything convertible to an integer. It is converted with
  # integer conversion; if that raises a type error (e.g. None) a ValueError is raised stating that
  # only integers are supported. Side effect: sets the global TensorFlow seed. Returns nothing.
  """Sets the seed for the random number generator.

  Uses `tf.set_random_seed`.

  Args:
    s: an integer.
  """
  try:
    s = int(s)
  except TypeError:
    # TODO(wangpeng): support this?
    raise ValueError(
        f'Argument `s` got an invalid value {s}. Only integers are supported.'
    )
  random_seed.set_seed(s)


@tf_export.tf_export('experimental.numpy.random.randn', v1=[])
@np_utils.np_doc('random.randn')
def randn(*args):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring claims the dtype is float64, but the dtype is whatever the numpy
  # dtype configuration gives for the Python float type (float32 by default unless the
  # allow-float64 option is enabled).
  # Draws standard normal samples. Input: variadic dimension sizes forming the output shape.
  # Returns a tensor of that shape by delegating to standard_normal with size set to the args.
  """Returns samples from a normal distribution.

  Uses `tf.random_normal`.

  Args:
    *args: The shape of the output array.

  Returns:
    An ndarray with shape `args` and dtype `float64`.
  """
  return standard_normal(size=args)


@tf_export.tf_export('experimental.numpy.random.standard_normal', v1=[])
@np_utils.np_doc('random.standard_normal')
def standard_normal(size=None):
  # [FUNCTION DOC]
  # Draws samples from the standard normal distribution. Input size: optional; nothing means a
  # scalar, an integer scalar means a one-dimensional shape of that length, otherwise a shape
  # sequence. The dtype is the numpy-config result type of the Python float. Returns a tensor of
  # normal samples; consumes global/graph random state (stateful op), raises nothing itself.
  # TODO(wangpeng): Use new stateful RNG
  if size is None:
    size = ()
  elif np_utils.isscalar(size):
    size = (size,)
  dtype = np_utils.result_type(float)
  return random_ops.random_normal(size, dtype=dtype)


@tf_export.tf_export('experimental.numpy.random.uniform', v1=[])
@np_utils.np_doc('random.uniform')
def uniform(low=0.0, high=1.0, size=None):
  # [FUNCTION DOC]
  # Draws samples uniformly from the interval between low and high. Inputs: low (default 0) and
  # high (default 1), scalars or array-likes, converted to arrays of the numpy-config float dtype;
  # size: optional output shape. If size is nothing, the shape is the broadcast of low and high
  # shapes. Returns a tensor of uniform samples; consumes random state.
  dtype = np_utils.result_type(float)
  low = np_array_ops.asarray(low, dtype=dtype)
  high = np_array_ops.asarray(high, dtype=dtype)
  if size is None:
    size = array_ops.broadcast_dynamic_shape(low.shape, high.shape)
  return random_ops.random_uniform(
      shape=size, minval=low, maxval=high, dtype=dtype
  )


@tf_export.tf_export('experimental.numpy.random.poisson', v1=[])
@np_utils.np_doc('random.poisson')
def poisson(lam=1.0, size=None):
  # [FUNCTION DOC]
  # Draws samples from a Poisson distribution. Inputs: lam, the rate (default 1); size: optional,
  # nothing means scalar shape, an integer scalar means a one-dimensional shape. Returns an
  # integer tensor (the numpy-config default integer dtype). Consumes random state.
  if size is None:
    size = ()
  elif np_utils.isscalar(size):
    size = (size,)
  return random_ops.random_poisson(shape=size, lam=lam, dtype=np_dtypes.int_)


@tf_export.tf_export('experimental.numpy.random.random', v1=[])
@np_utils.np_doc('random.random')
def random(size=None):
  # [FUNCTION DOC]
  # Returns uniform samples in the half-open range zero to one with the given optional size
  # (shape). Delegates to uniform with low zero and high one.
  return uniform(0.0, 1.0, size)


@tf_export.tf_export('experimental.numpy.random.rand', v1=[])
@np_utils.np_doc('random.rand')
def rand(*size):
  # [FUNCTION DOC]
  # Returns uniform samples between zero and one whose shape is given as variadic dimension
  # sizes. With no arguments the shape is empty (a scalar). Delegates to uniform.
  return uniform(0.0, 1.0, size)


@tf_export.tf_export('experimental.numpy.random.randint', v1=[])
@np_utils.np_doc('random.randint')
def randint(low, high=None, size=None, dtype=onp.int64):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: when high is omitted, the code sets high to low and low to zero, which matches
  # numpy's half-open convention; but low is forced to an integer even if high is a float, and
  # high is never converted or validated.
  # Draws random integers in the half-open range from low to high. Inputs: low (converted to
  # integer), optional high (if omitted the range is zero to low), size (nothing = scalar, an
  # integer = one-dimensional), dtype (default 64-bit integer). Raises ValueError if dtype, after
  # numpy-config conversion, is not 32- or 64-bit integer. Returns an integer tensor; consumes
  # random state.
  low = int(low)
  if high is None:
    high = low
    low = 0
  if size is None:
    size = ()
  elif isinstance(size, int):
    size = (size,)
  dtype_orig = dtype
  dtype = np_utils.result_type(dtype)
  accepted_dtypes = (onp.int32, onp.int64)
  if dtype not in accepted_dtypes:
    raise ValueError(
        f'Argument `dtype` got an invalid value {dtype_orig}. Only those '
        f'convertible to {accepted_dtypes} are supported.'
    )
  return random_ops.random_uniform(
      shape=size, minval=low, maxval=high, dtype=dtype
  )
