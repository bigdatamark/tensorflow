# [MODULE DOC]
# HELPER: helper module for tensor-valued random seeds used by datasets. Exposes one function
# wrapping the framework's seed derivation so that the results (and a tensor-valued input) are
# 64-bit integer tensors, avoiding the all-zero pair. No import-time side effects.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for generating Tensor-valued random seeds."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def get_seed(seed):
  # [FUNCTION DOC]
  # Computes the pair of local seeds for a dataset operation from an op-level seed. seed is an
  # integer, a 64-bit integer scalar tensor, or nothing/null. First asks the framework's seed
  # helper for the pair (which combines the graph-level seed with the op seed); a missing value
  # in either position becomes a constant zero tensor, other values become 64-bit integer
  # tensors. If both resulting seeds are zero, the second is replaced by the maximum 31-bit
  # value so the pair is never (0, 0), which would mean nondeterministic. Returns a pair of two
  # 64-bit integer scalar tensors. Adds ops to the current graph; no other side effects.
  """Returns the local seeds an operation should use given an op-specific seed.

  See `random_seed.get_seed` for more details. This wrapper adds support for
  the case where `seed` may be a tensor.

  Args:
    seed: An integer or a `tf.int64` scalar tensor.

  Returns:
    A tuple of two `tf.int64` scalar tensors that should be used for the local
    seed of the calling dataset.
  """
  seed, seed2 = random_seed.get_seed(seed)
  if seed is None:
    seed = constant_op.constant(0, dtype=dtypes.int64, name="seed")
  else:
    seed = ops.convert_to_tensor(seed, dtype=dtypes.int64, name="seed")
  if seed2 is None:
    seed2 = constant_op.constant(0, dtype=dtypes.int64, name="seed2")
  else:
    with ops.name_scope("seed2") as scope:
      seed2 = ops.convert_to_tensor(seed2, dtype=dtypes.int64)
      seed2 = array_ops.where_v2(
          math_ops.logical_and(
              math_ops.equal(seed, 0), math_ops.equal(seed2, 0)),
          constant_op.constant(2**31 - 1, dtype=dtypes.int64),
          seed2,
          name=scope)
  return seed, seed2
