# [MODULE DOC]
# Implementation module for the dataset random transformation: a source dataset producing an
# endless stream of pseudorandom 64-bit integers. Exposes a private factory function and a
# dataset source class, used by the main dataset module. No import-time side effects.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""The implementation of `tf.data.Dataset.random`."""

import warnings

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


def _random(  # pylint: disable=unused-private-name
    seed=None,
    rerandomize_each_iteration=None,
    name=None):
  # [FUNCTION DOC]
  # HELPER: factory used by the public dataset random method.
  # Inputs: seed (optional integer), rerandomize_each_iteration (optional boolean; reshuffle
  # the random sequence every epoch), name (optional string). Returns a new random dataset.
  """See `Dataset.random()` for details."""
  return _RandomDataset(
      seed=seed,
      rerandomize_each_iteration=rerandomize_each_iteration,
      name=name)


class _RandomDataset(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # Source dataset yielding pseudorandom scalar 64-bit integers. Holds the two derived seeds,
  # the rerandomize flag and the name. Created by the random factory function.
  """A `Dataset` of pseudorandom values."""

  def __init__(self, seed=None, rerandomize_each_iteration=None, name=None):
    # [FUNCTION DOC]
    # Inputs: seed (optional integer; combined with the global seed into two seeds),
    # rerandomize_each_iteration (optional boolean), name (optional string).
    # When rerandomizing is requested, uses the second-version random op with a dummy seed
    # generator; in the legacy graph mode (TF 1) it first emits a warning that this option only
    # works for repeat-based epochs. Otherwise uses the plain random op. Calls the parent
    # constructor with the resulting variant tensor.
    """A `Dataset` of pseudorandom values."""
    self._seed, self._seed2 = random_seed.get_seed(seed)
    self._rerandomize = rerandomize_each_iteration
    self._name = name
    if rerandomize_each_iteration:
      if not tf2.enabled():
        warnings.warn("In TF 1, the `rerandomize_each_iteration=True` option "
                      "is only supported for repeat-based epochs.")
      variant_tensor = ged_ops.random_dataset_v2(
          seed=self._seed,
          seed2=self._seed2,
          seed_generator=gen_dataset_ops.dummy_seed_generator(),
          rerandomize_each_iteration=self._rerandomize,
          **self._common_args)
    else:
      variant_tensor = ged_ops.random_dataset(
          seed=self._seed, seed2=self._seed2, **self._common_args)
    super().__init__(variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec: a scalar tensor spec of 64-bit integer type.
    return tensor_spec.TensorSpec([], dtypes.int64)
