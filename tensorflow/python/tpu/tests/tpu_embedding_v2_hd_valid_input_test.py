# [MODULE DOC]
# TPU test module for the mid level TPU embedding API (version 2) that exercises high dimensional
# (rank 3) dense, sparse and ragged inputs and per-feature output shape handling. It defines one
# test class derived from the shared embedding base test and runs under the TF test runner with
# V2 behavior enabled when executed as a script. It exposes no public API; importing it only
# defines the test class. It needs a TPU (or the base class strategy) to be meaningful.
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
"""Tests for TPU Embeddings mid level API on TPU."""


import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.platform import test
from tensorflow.python.tpu.tests import tpu_embedding_base_test


class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):
  # [CLASS DOC]
  # Test case covering the TPU embedding mid level API with high dimensional input features
  # (dense, sparse and ragged) and with explicit or inferred output shapes. All fixtures
  # (strategy, mid level API creation, dataset builders, batch sizes) come from the base class.

  def test_enqueue_dense_sparse_ragged(self):
    # [FUNCTION DOC]
    # Test: builds a strategy and an SGD based mid level embedding object, creates three
    # distributed iterators (high dimensional dense, sparse and ragged datasets, without fetching
    # to device), builds the embedding with three feature shapes of rank 3, then inside a
    # compiled function takes one element from each iterator, enqueues them for inference and
    # runs a dequeue on each replica. Passes if everything runs without raising; no values are
    # compared.
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    dataset = self._create_high_dimensional_dense_dataset(strategy)
    dense_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    sparse = self._create_high_dimensional_sparse_dataset(strategy)
    sparse_iter = iter(
        strategy.experimental_distribute_dataset(
            sparse,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    ragged = self._create_high_dimensional_ragged_dataset(strategy)
    ragged_iter = iter(
        strategy.experimental_distribute_dataset(
            ragged,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    mid_level_api.build([
        TensorShape([self.batch_size, self.data_batch_size, 1]),
        TensorShape([self.batch_size, self.data_batch_size, 2]),
        TensorShape([self.batch_size, self.data_batch_size, 3])
    ])

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Compiled body: enqueues one dense, one sparse and one ragged batch (taken from the
      # iterators) as non-training input, then runs the per-replica dequeue step and returns its
      # result. Takes no inputs; advances the three iterators as a side effect.

      def step():
        # [FUNCTION DOC]
        # Per-replica step: returns the dequeued embedding activations from the mid level API.
        return mid_level_api.dequeue()

      features = (next(dense_iter)[0], next(sparse_iter)[1],
                  next(ragged_iter)[2])
      mid_level_api.enqueue(features, training=False)
      return strategy.run(step)

    test_fn()

  def test_different_input_shapes(self):
    # [FUNCTION DOC]
    # Test: feeds one dense feature of shape one by three by one (built from a zeros constant,
    # unbatched, repeated and re-batched per replica) together with two sparse features, enqueues
    # them in a compiled function and dequeues. Then asserts that the mid level API recorded
    # output shapes of (1, 3) for the dense feature and (batch size, data batch size) for each
    # of the two sparse features.
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_high_dimensional_sparse_dataset(strategy)
    sparse_iter = iter(
        strategy.experimental_distribute_dataset(
            sparse,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))
    # Create a feature with shape (1, 3, 1)
    dense_feature = constant_op.constant(
        np.zeros(3), shape=(1, 3, 1), dtype=dtypes.int32)
    dense_dataset = dataset_ops.DatasetV2.from_tensors(
        dense_feature).unbatch().repeat().batch(
            1 * strategy.num_replicas_in_sync, drop_remainder=True)
    dense_iter = iter(
        strategy.experimental_distribute_dataset(
            dense_dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Compiled body: enqueues the dense feature and two sparse features as non-training input
      # and runs the per-replica dequeue, returning its result. Advances the iterators.

      def step():
        # [FUNCTION DOC]
        # Per-replica step: returns the dequeued embedding activations from the mid level API.
        return mid_level_api.dequeue()

      features = (next(dense_iter), next(sparse_iter)[1], next(sparse_iter)[2])
      mid_level_api.enqueue(features, training=False)
      return strategy.run(step)

    test_fn()

    self.assertEqual(mid_level_api._output_shapes, [
        TensorShape((1, 3)),
        TensorShape((self.batch_size, self.data_batch_size)),
        TensorShape((self.batch_size, self.data_batch_size))
    ])

  def test_output_shapes_priority_over_feature_config_and_build(self):
    # [FUNCTION DOC]
    # Test: sets explicit output shapes (2 by 4, for three features) directly on the mid level
    # API, then calls build with partially unknown shapes. Asserts the explicit output shapes are
    # kept unchanged, i.e. they take priority over shapes passed to build.
    _, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    # The output shapes setting in the feature config has the first priority.
    mid_level_api._output_shapes = [TensorShape((2, 4)) for _ in range(3)]
    mid_level_api.build([TensorShape((2, None, None)) for _ in range(3)])
    self.assertEqual(mid_level_api._output_shapes,
                     [TensorShape((2, 4)) for _ in range(3)])

if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
