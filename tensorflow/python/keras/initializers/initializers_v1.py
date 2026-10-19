# [MODULE DOC]
# Keras initializers for TF1 (graph-mode) behavior. Defines private aliases for the standard
# init_ops initializers and thin subclasses with Keras-style defaults (random normal, random
# uniform, truncated normal with small ranges; Lecun and He variance-scaling variants).
# Consumed by the initializers package registry when TF2 behavior is disabled. No import-time
# side effects beyond defining names.
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
"""Keras initializers for TF 1."""

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops


_v1_zeros_initializer = init_ops.Zeros
_v1_ones_initializer = init_ops.Ones
_v1_constant_initializer = init_ops.Constant
_v1_variance_scaling_initializer = init_ops.VarianceScaling
_v1_orthogonal_initializer = init_ops.Orthogonal
_v1_identity = init_ops.Identity
_v1_glorot_uniform_initializer = init_ops.GlorotUniform
_v1_glorot_normal_initializer = init_ops.GlorotNormal


class RandomNormal(init_ops.RandomNormal):
  # [CLASS DOC]
  # Keras version of the random normal initializer for TF1: same as the underlying init_ops class
  # but with a default standard deviation of 0.05.

  def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=dtypes.float32):
    # [FUNCTION DOC]
    # Constructor. Inputs: mean (default 0.0), standard deviation (default 0.05), optional random
    # seed (default none) and data type (default float32). Passes them to the parent class.
    super(RandomNormal, self).__init__(
        mean=mean, stddev=stddev, seed=seed, dtype=dtype)


class RandomUniform(init_ops.RandomUniform):
  # [CLASS DOC]
  # Keras version of the random uniform initializer for TF1: same as the underlying init_ops
  # class but the default range is from minus 0.05 to 0.05.

  def __init__(self, minval=-0.05, maxval=0.05, seed=None,
               dtype=dtypes.float32):
    # [FUNCTION DOC]
    # Constructor. Inputs: lower bound (default minus 0.05), upper bound (default 0.05), optional
    # random seed and data type (default float32). Passes them to the parent class.
    super(RandomUniform, self).__init__(
        minval=minval, maxval=maxval, seed=seed, dtype=dtype)


class TruncatedNormal(init_ops.TruncatedNormal):
  # [CLASS DOC]
  # Keras version of the truncated normal initializer for TF1: same as the underlying init_ops
  # class but with a default standard deviation of 0.05.

  def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=dtypes.float32):
    # [FUNCTION DOC]
    # Constructor. Inputs: mean (default 0.0), standard deviation (default 0.05), optional random
    # seed (default none) and data type (default float32). Passes them to the parent class.
    super(TruncatedNormal, self).__init__(
        mean=mean, stddev=stddev, seed=seed, dtype=dtype)


class LecunNormal(init_ops.VarianceScaling):
  # [CLASS DOC]
  # Lecun normal initializer: variance scaling with scale one, fan-in mode and truncated normal
  # distribution. Only the seed is configurable and serialized.

  def __init__(self, seed=None):
    # [FUNCTION DOC]
    # Constructor. Input: optional random seed. Configures the parent variance-scaling
    # initializer with this class's fixed scale, mode and distribution.
    super(LecunNormal, self).__init__(
        scale=1., mode='fan_in', distribution='truncated_normal', seed=seed)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping containing only the seed, so the fixed scale, mode and distribution are
    # not serialized.
    return {'seed': self.seed}


class LecunUniform(init_ops.VarianceScaling):
  # [CLASS DOC]
  # Lecun uniform initializer: variance scaling with scale one, fan-in mode and uniform
  # distribution. Only the seed is configurable and serialized.

  def __init__(self, seed=None):
    # [FUNCTION DOC]
    # Constructor. Input: optional random seed. Configures the parent variance-scaling
    # initializer with this class's fixed scale, mode and distribution.
    super(LecunUniform, self).__init__(
        scale=1., mode='fan_in', distribution='uniform', seed=seed)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping containing only the seed, so the fixed scale, mode and distribution are
    # not serialized.
    return {'seed': self.seed}


class HeNormal(init_ops.VarianceScaling):
  # [CLASS DOC]
  # He normal initializer: variance scaling with scale two, fan-in mode and truncated normal
  # distribution. Only the seed is configurable and serialized.

  def __init__(self, seed=None):
    # [FUNCTION DOC]
    # Constructor. Input: optional random seed. Configures the parent variance-scaling
    # initializer with this class's fixed scale, mode and distribution.
    super(HeNormal, self).__init__(
        scale=2., mode='fan_in', distribution='truncated_normal', seed=seed)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping containing only the seed, so the fixed scale, mode and distribution are
    # not serialized.
    return {'seed': self.seed}


class HeUniform(init_ops.VarianceScaling):
  # [CLASS DOC]
  # He uniform initializer: variance scaling with scale two, fan-in mode and uniform
  # distribution. Only the seed is configurable and serialized.

  def __init__(self, seed=None):
    # [FUNCTION DOC]
    # Constructor. Input: optional random seed. Configures the parent variance-scaling
    # initializer with this class's fixed scale, mode and distribution.
    super(HeUniform, self).__init__(
        scale=2., mode='fan_in', distribution='uniform', seed=seed)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping containing only the seed, so the fixed scale, mode and distribution are
    # not serialized.
    return {'seed': self.seed}
