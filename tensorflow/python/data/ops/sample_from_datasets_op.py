# [MODULE DOC]
# Implementation module for the dataset sample-from-datasets transformation: builds a dataset
# that interleaves elements of several input datasets, choosing the source of each element at
# random according to given weights (or uniformly). It builds a selector dataset of random
# choices and delegates to the directed interleave implementation. No import-time side effects.
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
"""The implementation of `tf.data.Dataset.sample_from_datasets`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import directed_interleave_op
from tensorflow.python.data.ops import map_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.types import data as data_types


def _sample_from_datasets(datasets,  # pylint: disable=unused-private-name
                          weights=None,
                          seed=None,
                          stop_on_empty_dataset=False,
                          rerandomize_each_iteration=None):
  # [FUNCTION DOC]
  # HELPER: implementation behind the public sample-from-datasets method.
  # Inputs: datasets (non-empty list of datasets), weights (optional: nothing for uniform choice,
  # a list or tensor of floating-point weights with one per dataset, or a dataset yielding weight
  # vectors), seed (optional integer), stop_on_empty_dataset (boolean, default false; stop when a
  # selected input is exhausted), rerandomize_each_iteration (optional boolean for the random
  # selector stream).
  # Raises a value error for empty datasets, a weights shape or length mismatch, and a type error
  # when weights are not 32- or 64-bit floating point. For list weights, datasets with zero weight
  # are dropped (if all are zero, a single empty dataset is used). When only one dataset remains
  # and weights are not a dataset, it is returned directly. Otherwise a selector dataset of random
  # indices drawn from log-probabilities is built and a directed interleave dataset is returned.
  """See `Dataset.sample_from_datasets()` for details."""

  def _skip_datasets_with_zero_weight(datasets, weights):
    # [FUNCTION DOC]
    # HELPER: nested helper filtering out datasets whose weight is not positive. Returns paired
    # sequences of the remaining datasets and weights; if none remain, returns a single empty
    # dataset (first dataset limited to zero elements) with weight one.
    datasets_and_weights = [(dataset, weight)
                            for (dataset, weight) in zip(datasets, weights)
                            if weight > 0]
    return (zip(*datasets_and_weights) if datasets_and_weights else
            ([datasets[0].take(0)], [1.]))

  if not datasets:
    raise ValueError("Invalid `datasets`. `datasets` should not be empty.")

  if not isinstance(weights, data_types.DatasetV2):
    if weights is None:
      # Select inputs with uniform probability.
      logits = [[1.0] * len(datasets)]

    else:
      if isinstance(weights, tensor.Tensor):
        if not weights.shape.is_compatible_with([len(datasets)]):
          raise ValueError(f"Invalid `weights`. The shape of `weights` "
                           f"should be compatible with `[len(datasets)]` "
                           f"but is {weights.shape}.")
      else:
        if len(datasets) != len(weights):
          raise ValueError(f"Invalid `weights`. `weights` should have the "
                           f"same length as `datasets` but got "
                           f"`len(weights)={len(weights)}` vs. "
                           f"`len(datasets)={len(datasets)}`.")

      # Use the given `weights` as the probability of choosing the respective
      # input.
      if not isinstance(weights, tensor.Tensor):
        datasets, weights = _skip_datasets_with_zero_weight(datasets, weights)
      weights = ops.convert_to_tensor(weights, name="weights")
      if weights.dtype not in (dtypes.float32, dtypes.float64):
        raise TypeError(f"Invalid `weights`. `weights` type must be either "
                        f"`tf.float32` or `tf.float64` but is "
                        f"{weights.dtype}.")

      # The `stateless_multinomial()` op expects log-probabilities, as opposed
      # to weights.
      logits = array_ops.expand_dims(math_ops.log(weights, name="logits"), 0)

    # NOTE(mrry): We only specialize when `weights` is not a `Dataset`. When
    # it is a `Dataset`, it is possible that evaluating it has a side effect
    # the user depends on.
    if len(datasets) == 1:
      return datasets[0]

    def select_dataset_constant_logits(seed):
      # [FUNCTION DOC]
      # HELPER: nested map callback. Given a pair of random seed values, draws one index from the
      # fixed log-probabilities of the enclosing scope using a stateless multinomial op and
      # returns it as a scalar.
      return array_ops.squeeze(
          gen_stateless_random_ops.stateless_multinomial(
              logits, 1, seed=seed),
          axis=[0, 1])

    selector_input = map_op._MapDataset(  # pylint: disable=protected-access
        dataset_ops.Dataset.random(
            seed=seed,
            rerandomize_each_iteration=rerandomize_each_iteration).batch(2),
        select_dataset_constant_logits,
        use_inter_op_parallelism=False)

  else:  # isinstance(weights, DatasetV2)
    # Use each element of the given `weights` dataset as the probability of
    # choosing the respective input.
    #
    # The `stateless_multinomial()` op expects log-probabilities, as opposed
    # to weights.
    logits_ds = weights.map(lambda *p: math_ops.log(p, name="logits"))

    def select_dataset_varying_logits(logits, seed):
      # [FUNCTION DOC]
      # HELPER: nested map callback. Given log-probabilities for this step and a pair of random
      # seed values, draws one index with a stateless multinomial op and returns it as a scalar.
      return array_ops.squeeze(
          gen_stateless_random_ops.stateless_multinomial(
              logits, 1, seed=seed),
          axis=[0, 1])

    logits_and_seeds = dataset_ops.Dataset.zip(
        (logits_ds,
         dataset_ops.Dataset.random(
             seed=seed,
             rerandomize_each_iteration=rerandomize_each_iteration).batch(2)))
    selector_input = map_op._MapDataset(  # pylint: disable=protected-access
        logits_and_seeds,
        select_dataset_varying_logits,
        use_inter_op_parallelism=False)

  return directed_interleave_op._directed_interleave(  # pylint: disable=protected-access
      selector_input, datasets, stop_on_empty_dataset
  )
