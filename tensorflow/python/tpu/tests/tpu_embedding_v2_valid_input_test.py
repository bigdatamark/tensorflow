# [MODULE DOC]
# TPU test module for the mid level TPU embedding API (version 2) with valid inputs: passing
# nothing as gradients, mixing sparse and ragged inputs, per-device enqueue, weighted inputs,
# deterministic configuration generation and features with missing samples. Defines one test
# class on top of the shared embedding base test; runs under the TF test runner with V2 behavior
# when executed as a script. No public API.
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
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test
from tensorflow.python.util import nest


class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):
  # [CLASS DOC]
  # Test case of positive scenarios for the TPU embedding mid level API with valid sparse,
  # ragged and weighted inputs. Strategy, dataset builders, tables and replica helpers come from
  # the shared base test class.

  def test_pass_none_to_apply_gradients(self):
    # [FUNCTION DOC]
    # Test (skipped in the open source build): builds an SGD mid level API for three features and
    # a sparse dataset. One compiled function enqueues, dequeues and applies gradients that are
    # nothing for the first two features and ones for the third; another only enqueues and
    # dequeues without training. Compares the second lookup to the first minus the expected SGD
    # update (learning rate 0.1, scaled by the replica count, with a third of it for the mean
    # combined row), asserting that the nothing gradients caused no change.
    self.skip_if_oss()
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    mid_level_api.build([
        TensorShape((self.batch_size, 2)),
        TensorShape((self.batch_size, 2)),
        TensorShape((self.batch_size, 3))
    ])
    dataset = self._create_sparse_dataset(strategy)
    data = next(
        iter(
            strategy.experimental_distribute_dataset(
                dataset,
                options=distribute_lib.InputOptions(
                    experimental_fetch_to_device=False))))

    @def_function.function
    def embedding_and_set_gradients(data):
      # [FUNCTION DOC]
      # Compiled body: enqueues data (training mode default) and runs the per-replica function
      # on the strategy, returning the activations produced before the gradient update.
      mid_level_api.enqueue(data)
      def tpu_fn():
        # [FUNCTION DOC]
        # Per-replica function: dequeues activations, applies gradients that are nothing for the
        # first two features and a tensor of ones shaped like the third activation, and returns
        # the dequeued activations.
        results = mid_level_api.dequeue()
        mid_level_api.apply_gradients((None, None,
                                       array_ops.ones_like(results[2])))
        return results
      return strategy.run(tpu_fn)

    @def_function.function
    def embedding_only(data):
      # [FUNCTION DOC]
      # Compiled body: enqueues data in non-training mode and runs the per-replica dequeue on the
      # strategy, returning the activations.
      mid_level_api.enqueue(data, training=False)
      def tpu_fn():
        # [FUNCTION DOC]
        # Per-replica function: returns the dequeued activations of the mid level API.
        return mid_level_api.dequeue()
      return strategy.run(tpu_fn)

    first = self._get_replica_numpy(
        embedding_and_set_gradients(data), strategy, 0)
    second = self._get_replica_numpy(embedding_only(data), strategy, 0)

    # First two features should be the same as None gradient was applied.
    # Third feature had gradient of 1 passed in from each core.
    # Each core received the same ids per core and returned the following batch:
    # [ row 3, row 0 + row 1 + row 2 ]
    # so gradient update was (learning rate = 0.1):
    #   row 0: -1/3*0.1
    #   row 1: -1/3*0.1
    #   row 2: -1/3*0.1
    #   row 3: -1*0.1
    # There is a factor of num_replicas because each replica gave an update.

    num_replicas = strategy.num_replicas_in_sync
    update = ([[0.0]], [[0.0]],
              [[0.1 * num_replicas], [0.1 / 3 * num_replicas]])
    golden = tuple([feature-np.array(up) for feature, up in zip(first, update)])

    self.assertAllClose(golden, second)

  def test_enqueue_sparse_and_ragged(self):
    # [FUNCTION DOC]
    # Test (skipped in the open source build): distributes a sparse and a ragged dataset and, in
    # a compiled function, enqueues a mixed structure (sparse for features zero and two, ragged
    # for feature one) for inference, then dequeues. Passes if no error is raised.
    self.skip_if_oss()
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_sparse_dataset(strategy)
    ragged = self._create_ragged_dataset(strategy)
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
    def test_fn():
      # [FUNCTION DOC]
      # Compiled body: takes one sparse and one ragged batch, enqueues a mix of both and runs the
      # per-replica dequeue, returning the result.
      def step():
        # [FUNCTION DOC]
        # Per-replica step: returns the dequeued activations of the mid level API.
        return mid_level_api.dequeue()

      sparse_features = next(sparse_iter)
      ragged_features = next(ragged_iter)
      features = (sparse_features[0], ragged_features[1], sparse_features[2])
      mid_level_api.enqueue(features, training=False)
      return strategy.run(step)

    test_fn()

  def test_enqueue_per_device(self):
    # [FUNCTION DOC]
    # Test (skipped in the open source build): in a compiled function, first enqueues a sparse
    # batch globally and dequeues activations; then enqueues the per-core local slices of the
    # same batch to each core's own device through a distributed-values function and dequeues
    # again. Asserts the replica zero activations of both paths are close, then runs the
    # compiled function a second time to check it can be re-executed.
    self.skip_if_oss()
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_sparse_dataset(strategy)
    sparse_iter = iter(
        strategy.experimental_distribute_dataset(
            sparse,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Compiled body: enqueues the sparse batch globally and runs the activation getter; then
      # builds distributed values by enqueueing each core's local slice to its device and runs the
      # getter again. Returns the global activations, the per-device activations and the two dense
      # pass-through values.
      def get_activations(dense_value):
        # [FUNCTION DOC]
        # Per-replica function: returns a pair of the dequeued activations and the dense value
        # passed in (used to force the enqueue function to execute).
        return mid_level_api.dequeue(), dense_value

      sparse_features = next(sparse_iter)
      mid_level_api.enqueue(sparse_features, training=False)
      activations, dense_value1 = strategy.run(get_activations, args=(0.0,))

      def enqueue_fn(ctx):
        # [FUNCTION DOC]
        # Value function run per replica with the input context: finds the replica's core index
        # and device, slices out that core's local sparse features, enqueues them (non-training)
        # to that device, and returns the constant zero as a placeholder dense value.
        core_id = ctx.replica_id_in_sync_group
        device = strategy.extended.worker_devices[core_id]
        sparse_features_local = nest.map_structure(
            lambda x: strategy.experimental_local_results(x)[core_id],
            sparse_features)
        mid_level_api.enqueue(sparse_features_local, training=False,
                              device=device)
        return 0.0

      data = strategy.experimental_distribute_values_from_function(
          enqueue_fn)
      per_device_activations, dense_value2 = strategy.run(get_activations,
                                                          args=(data,))
      return activations, per_device_activations, dense_value1, dense_value2

    activations, per_device_activations, _, _ = test_fn()

    # Extact per core numpy arrays and check that both sparse and ragged have
    # the same results.
    activations0 = self._get_replica_numpy(activations, strategy, 0)
    per_device_activations0 = self._get_replica_numpy(
        per_device_activations, strategy, 0)
    self.assertAllClose(activations0, per_device_activations0)
    test_fn()

  @parameterized.parameters(True, False)
  def test_enqueue_with_weights(self, ragged):
    # [FUNCTION DOC]
    # Parameterized test over ragged (true: ragged dataset, false: sparse dataset, which also
    # triggers an explicit build with fixed shapes). Uses a weight of 0.5. Gets activations with
    # no weights and with weights (the second feature's weight replaced by nothing) via a compiled
    # function. Asserts the weighted result equals the unweighted one scaled by 0.5, 1.0 and 1.0
    # for the three features (the third table uses a mean combiner, so equal weights cancel).
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    weight = 0.5
    if ragged:
      dataset = self._create_ragged_dataset(strategy, include_weights=True,
                                            weight=weight)
    else:
      dataset = self._create_sparse_dataset(strategy, include_weights=True,
                                            weight=weight)
      mid_level_api.build([
          TensorShape((self.batch_size, 2)),
          TensorShape((self.batch_size, 2)),
          TensorShape((self.batch_size, 3))
      ])

    dataset_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def enqueue_and_get(features, weights):
      # [FUNCTION DOC]
      # Compiled body: enqueues the features with the given optional weights in non-training mode
      # and runs the per-replica dequeue, returning the activations.
      def get_activations():
        # [FUNCTION DOC]
        # Per-replica function: returns the dequeued activations of the mid level API.
        return mid_level_api.dequeue()
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(get_activations)

    features, weights = next(dataset_iter)
    # Replace the weight for the second feature by None to test.
    weights = (weights[0], None, weights[2])

    no_weights_activations = enqueue_and_get(features, weights=None)
    weights_activations = enqueue_and_get(features, weights=weights)

    # Extact per core numpy arrays.
    no_weights0 = self._get_replica_numpy(no_weights_activations, strategy, 0)
    weights0 = self._get_replica_numpy(weights_activations, strategy, 0)
    # videos table has sum combiner and users table has mean combiner.
    # i.e. users table lookups isn't affected by the weights as all the weights
    # are the same.
    # Tuple entry 0 and 1 are the watched and favorited features from the videos
    # table and entry 2 is the friends feature from the users table.
    # Note that None was passed as a weight for entry 1 so weight should have no
    # effect.
    weight = (0.5, 1.0, 1.0)
    golden = tuple([no_weight * w for no_weight, w in zip(no_weights0, weight)])

    self.assertAllClose(golden, weights0)

  def test_same_config_different_instantiations(self):
    # [FUNCTION DOC]
    # Test (skipped in the open source build): draws random dimensions and vocabulary sizes for
    # thirty tables, builds the embedding configuration protocol buffer twice from the same data
    # and asserts both are equal, i.e. config generation is deterministic.
    self.skip_if_oss()
    num_tables = 30
    table_dim = np.random.randint(1, 128, size=[num_tables])
    table_vocab_size = np.random.randint(100, 1000, size=[num_tables])
    table_names = ['table{}'.format(i) for i in range(num_tables)]
    table_data = list(zip(table_dim, table_vocab_size, table_names))
    strategy = self._get_strategy()

    def tpu_embedding_config():
      # [FUNCTION DOC]
      # Creates one feature config per table entry (zero initializer), a mid level embedding under
      # the strategy scope with an Adagrad optimizer, sets output shapes to a vector of 128 per
      # feature, and returns the generated configuration protocol buffer. Takes no inputs.
      feature_configs = []
      for dim, vocab, name in table_data:
        feature_configs.append(tpu_embedding_v2_utils.FeatureConfig(
            table=tpu_embedding_v2_utils.TableConfig(
                vocabulary_size=int(vocab), dim=int(dim),
                initializer=init_ops_v2.Zeros(), name=name)))
      optimizer = tpu_embedding_v2_utils.Adagrad(
          learning_rate=0.1)
      with strategy.scope():
        mid_level_api = tpu_embedding_v2.TPUEmbedding(
            feature_config=feature_configs,
            optimizer=optimizer)
      mid_level_api._output_shapes = [TensorShape(128)] * len(feature_configs)
      return mid_level_api._create_config_proto()

    self.assertProtoEquals(tpu_embedding_config(), tpu_embedding_config())

  @parameterized.parameters([True, False])
  def test_missing_feature(self, is_sparse):
    # [FUNCTION DOC]
    # Parameterized test over is_sparse. Creates a single watched feature embedding, builds
    # sparse or ragged input in which the last sample is missing, turns it into a repeating
    # batched distributed dataset, and enqueues and dequeues one batch in a compiled function.
    # Passes if no error is raised; no values are compared.
    strategy = self._get_strategy()
    with strategy.scope():
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      mid_level_api = tpu_embedding_v2.TPUEmbedding(
          feature_config=tpu_embedding_v2_utils.FeatureConfig(
              table=self.table_video, name='watched'),
          optimizer=optimizer)
    # Create sparse or ragged feature with last sample missing.
    if is_sparse:
      features = sparse_tensor.SparseTensor(
          indices=self.feature_watched_indices[:-1],
          values=self.feature_watched_values[:-1],
          dense_shape=[self.data_batch_size, 2])
    else:
      features = ragged_tensor.RaggedTensor.from_row_lengths(
          row_lengths=[1, 2, 2, 0], values=self.feature_watched_values[:-1])

    dataset = dataset_ops.DatasetV2.from_tensors(features)

    dataset = dataset.unbatch().repeat().batch(
        self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)
    dataset_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Compiled body: enqueues the next batch in non-training mode and runs the per-replica
      # dequeue, returning the activations.

      def get_activations():
        # [FUNCTION DOC]
        # Per-replica function: returns the dequeued activations of the mid level API.
        return mid_level_api.dequeue()

      mid_level_api.enqueue(next(dataset_iter), training=False)
      return strategy.run(get_activations)

    test_fn()

if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
