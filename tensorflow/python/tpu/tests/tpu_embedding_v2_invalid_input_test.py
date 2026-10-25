# [MODULE DOC]
# TPU test module for the mid level TPU embedding API (version 2) that checks error handling for
# invalid inputs: duplicate table names, non-tensor or mis-structured gradients, weights supplied
# for dense input, weights of the wrong tensor kind, mismatched feature or weight structures, and
# CPU-resident tensors fed to enqueue. Defines one test class on top of the shared embedding base
# test; runs under the TF test runner with V2 behavior when executed as a script. No public API.
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
from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test


class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):
  # [CLASS DOC]
  # Test case collecting negative tests for the TPU embedding mid level API: each test provokes
  # an invalid usage and asserts that the expected error type and message are raised. Strategy,
  # dataset and mid level creation helpers come from the shared base test class.

  def test_tables_with_same_name(self):
    # [FUNCTION DOC]
    # Test: under a TPU strategy scope, constructs the embedding object with two feature configs
    # whose tables (both small, same dimension) share the name table. Asserts a value error
    # stating that multiple tables with that name were found.
    with self.assertRaisesRegex(
        ValueError, 'Multiple tables with name table found.'):
      with self._get_strategy().scope():
        tpu_embedding_v2.TPUEmbedding(
            (tpu_embedding_v2_utils.FeatureConfig(
                table=tpu_embedding_v2_utils.TableConfig(
                    name='table',
                    vocabulary_size=4,
                    dim=2,
                    initializer=self.initializer,),
                name='watched'),
             tpu_embedding_v2_utils.FeatureConfig(
                 table=tpu_embedding_v2_utils.TableConfig(
                     name='table',
                     vocabulary_size=4,
                     dim=2,
                     initializer=self.initializer),
                 name='favorited')),
            tpu_embedding_v2_utils.SGD(learning_rate=0.1))

  def test_pass_non_tensor_to_apply_gradients(self):
    # [FUNCTION DOC]
    # Test (skipped in the open source build): builds an SGD mid level API, then runs two
    # compiled functions on the strategy. The first applies gradients given as a tuple of plain
    # integers and must raise a value error mentioning a non-tensor type. The second applies a
    # list of integers where a tuple of three is expected and must raise a type error about
    # differing nested structures.
    self.skip_if_oss()
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    # We aren't going to actually run anything, so the batch_size here does not
    # matter.
    mid_level_api.build(64)

    # Test pass non tensor to apply_gradients.
    @def_function.function
    def test_apply_1():
      # [FUNCTION DOC]
      # Compiled body: calls apply gradients with a tuple of plain integers instead of tensors;
      # expected to fail with a value error. Returns nothing.
      mid_level_api.apply_gradients((1, 2, 3))

    with self.assertRaisesRegex(ValueError, 'found non-tensor type'):
      strategy.run(test_apply_1)

    # Test pass different structure to apply_gradients.
    @def_function.function
    def test_apply_2():
      # [FUNCTION DOC]
      # Compiled body: calls apply gradients with a list rather than the expected tuple of three
      # entries; expected to fail with a type error about nested structure. Returns nothing.
      # This should be a tuple as feature_config is a tuple of 3 configs.
      mid_level_api.apply_gradients([1, 2, 3])

    with self.assertRaisesRegex(
        TypeError, 'The two structures don\'t have the same nested structure.'):
      strategy.run(test_apply_2)

  def test_enqueue_weight_for_dense_tensor(self):
    # [FUNCTION DOC]
    # Test: distributes a dense dataset that includes weights and, in a compiled function,
    # enqueues the dense features together with weights. Asserts a value error stating that a
    # weight was specified for dense input.
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    dataset = self._create_dense_dataset(strategy, include_weights=True)
    dense_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Compiled body: takes features and weights from the dense iterator, enqueues both
      # (non-training) and runs the per-replica dequeue; returns its result. The enqueue is what
      # is expected to raise.
      def step():
        # [FUNCTION DOC]
        # Per-replica step: returns the dequeued activations of the mid level API.
        return mid_level_api.dequeue()

      features, weights = next(dense_iter)
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(step)

    with self.assertRaisesRegex(ValueError, 'Weight specified for dense input'):
      test_fn()

  def test_enqueue_wrong_weight_type_for_sparse_and_ragged_tensor(self):
    # [FUNCTION DOC]
    # Test (skipped in the open source build): creates sparse and ragged datasets with weights.
    # First enqueues sparse features with ragged weights and asserts a value error saying the
    # weight type does not match the sparse input type; then enqueues ragged features with sparse
    # weights and asserts the analogous error for ragged input.
    self.skip_if_oss()
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_sparse_dataset(strategy, include_weights=True)
    ragged = self._create_ragged_dataset(strategy, include_weights=True)
    sparse_iter = iter(
        strategy.experimental_distribute_dataset(
            sparse,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))
    ragged_iter = iter(
        strategy.experimental_distribute_dataset(
            ragged,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def test_sparse_fn():
      # [FUNCTION DOC]
      # Compiled body: enqueues sparse features paired with ragged weights (mismatched kinds),
      # then runs the dequeue step; the enqueue is expected to raise a value error.
      def step():
        # [FUNCTION DOC]
        # Per-replica step: returns the dequeued activations of the mid level API.
        return mid_level_api.dequeue()

      features, _ = next(sparse_iter)
      _, weights = next(ragged_iter)
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(step)

    with self.assertRaisesRegex(
        ValueError, 'which does not match type input which is SparseTensor.'):
      test_sparse_fn()

    @def_function.function
    def test_ragged_fn():
      # [FUNCTION DOC]
      # Compiled body: enqueues ragged features paired with sparse weights (mismatched kinds),
      # then runs the dequeue step; the enqueue is expected to raise a value error.
      def step():
        # [FUNCTION DOC]
        # Per-replica step: returns the dequeued activations of the mid level API.
        return mid_level_api.dequeue()

      _, weights = next(sparse_iter)
      features, _ = next(ragged_iter)
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(step)

    with self.assertRaisesRegex(
        ValueError, 'which does not match type input which is RaggedTensor.'):
      test_ragged_fn()

  def test_enqueue_incorrect_structure_for_features_and_weights(self):
    # [FUNCTION DOC]
    # Test (skipped in the open source build): enqueues sparse features cut down to a one-element
    # tuple, and separately weights cut down to a one-element tuple, each inside a compiled
    # function. Asserts that both cases raise a value error (coming from the nested structure
    # comparison against the feature configuration).
    self.skip_if_oss()
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_sparse_dataset(strategy, include_weights=True)
    sparse_iter = iter(
        strategy.experimental_distribute_dataset(
            sparse,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def test_features_fn():
      # [FUNCTION DOC]
      # Compiled body: enqueues a features structure truncated to one element (wrong structure)
      # and runs the dequeue step; expected to raise a value error at enqueue.
      def step():
        # [FUNCTION DOC]
        # Per-replica step: returns the dequeued activations of the mid level API.
        return mid_level_api.dequeue()

      features = next(sparse_iter)
      features = (features[0],)
      mid_level_api.enqueue(features, training=False)
      return strategy.run(step)

    # The error here is raised from nest.assert_same_structure
    with self.assertRaises(ValueError):
      test_features_fn()

    @def_function.function
    def test_weights_fn():
      # [FUNCTION DOC]
      # Compiled body: enqueues valid features with a weights structure truncated to one element
      # (wrong structure) and runs the dequeue step; expected to raise a value error at enqueue.
      def step():
        # [FUNCTION DOC]
        # Per-replica step: returns the dequeued activations of the mid level API.
        return mid_level_api.dequeue()

      features, weights = next(sparse_iter)
      weights = (weights[0],)
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(step)

    # The error here is raised from nest.assert_same_structure
    with self.assertRaises(ValueError):
      test_weights_fn()

  def test_enqueue_cpu_tensor(self):
    # [FUNCTION DOC]
    # Test: distributes a dense dataset with default options (tensors stay on the TPU input
    # device) and enqueues its features inside a compiled function, outside of the per-replica
    # computation. Asserts a value error that the tensor is on a TPU input device.
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    dataset = self._create_dense_dataset(strategy)
    dense_iter = iter(strategy.experimental_distribute_dataset(dataset))

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Compiled body: takes one dense batch, enqueues it as non-training input, runs the
      # activation getter on every replica and returns the activations. The enqueue is the
      # expected failure point.
      def get_activations():
        # [FUNCTION DOC]
        # Per-replica step: returns the dequeued activations of the mid level API.
        return mid_level_api.dequeue()

      features = next(dense_iter)
      mid_level_api.enqueue(features, training=False)
      activations = strategy.run(get_activations)
      return activations

    with self.assertRaisesRegex(ValueError, 'which is on a TPU input device'):
      test_fn()

  @parameterized.parameters([True, False])
  def test_enqueue_cpu_tensor_with_outside_compilation(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized test over use_mlir (true or false). When true, enables the MLIR bridge
    # globally (side effect on process-wide config). Distributes a dense dataset with default
    # options and, in a compiled function, runs a per-replica function that enqueues the
    # replica's features and dequeues. Asserts a value error that the tensor is on a TPU input
    # device.

    if use_mlir:
      config.enable_mlir_bridge()

    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    dataset = self._create_dense_dataset(strategy)
    dense_iter = iter(strategy.experimental_distribute_dataset(dataset))

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Compiled body: runs the per-replica function on the strategy with the next distributed
      # dense batch as its argument and returns the activations. Expected to raise on enqueue.
      def get_activations(features):
        # [FUNCTION DOC]
        # Per-replica function: enqueues the given features as non-training input and returns the
        # dequeued activations of the mid level API.
        mid_level_api.enqueue(features, training=False)
        return mid_level_api.dequeue()

      activations = strategy.run(get_activations, args=(next(dense_iter),))
      return activations

    with self.assertRaisesRegex(ValueError, 'which is on a TPU input device'):
      test_fn()

if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
