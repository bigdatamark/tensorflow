# [MODULE DOC]
# Test module for numerical correctness of the V0 TPU embedding mid-level API. Currently contains a
# test that checks weighted lookups against unweighted ones. Needs a TPU; runs with v2 behavior
# through the test main entry. Builds on the shared base test module.
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
"""Tests for TPU Embeddings mid level API on TPU."""

from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v1
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test


class TPUEmbeddingV0CorrectnessTest(tpu_embedding_base_test.TPUEmbeddingBaseTest
                                   ):
  # [CLASS DOC]
  # Correctness test case for the V0 embedding API, derived from the shared TPU embedding base test.
  # Overrides the strategy getter (cached strategy), the mid-level factory (V0 object) and the
  # combined strategy-and-embedding helper.

  def _get_strategy(self):
    # [FUNCTION DOC]
    # HELPER: strategy getter override. Returns the cached strategy attribute if the instance has one,
    # otherwise delegates to the base class which initializes the TPU and builds a new strategy.
    if hasattr(self, 'strategy'):
      return self.strategy
    return super(TPUEmbeddingV0CorrectnessTest, self)._get_strategy()

  def _create_mid_level(self, optimizer=None):
    # [FUNCTION DOC]
    # HELPER: factory override. Input: optimizer, optional (default gradient descent, learning rate 0.1).
    # Returns a V0 TPU embedding built from the feature config and the optimizer.
    # Create `TPUEmbedding` object.
    if optimizer is None:
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)

    return tpu_embedding_v1.TPUEmbeddingV0(
        feature_config=self.feature_config, optimizer=optimizer)

  def _create_strategy_and_mid_level(self, optimizer_name):
    # [FUNCTION DOC]
    # HELPER: creates the strategy and the V0 embedding for a named optimizer.
    # KNOWN ISSUE: unlike the base class version this returns only a pair of strategy and embedding
    # object (no optimizer) and does not support the adagrad-momentum name; callers must use the pair form.
    # Input: optimizer_name, one of sgd, adagrad, adam, ftrl (learning rate 0.1). Gets the strategy,
    # builds the optimizer and the embedding object in its scope. Raises a value error for other names.
    strategy = self._get_strategy()

    with strategy.scope():
      if optimizer_name == 'sgd':
        embedding_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      elif optimizer_name == 'adagrad':
        embedding_optimizer = tpu_embedding_v2_utils.Adagrad(
            learning_rate=0.1)
      elif optimizer_name == 'adam':
        embedding_optimizer = tpu_embedding_v2_utils.Adam(
            learning_rate=0.1)
      elif optimizer_name == 'ftrl':
        embedding_optimizer = tpu_embedding_v2_utils.FTRL(
            learning_rate=0.1)
      else:
        raise ValueError('optimizer is not recognized: ', optimizer_name)

      mid_level_api = self._create_mid_level(optimizer=embedding_optimizer)

    return strategy, mid_level_api

  @parameterized.parameters(True, False)
  def test_enqueue_with_weights(self, ragged):
    # [FUNCTION DOC]
    # Parameterized over ragged (true or false) inputs. Scenario: lookups with and without per-feature
    # weights. Creates a gradient-descent setup and a dataset (ragged or sparse) with weights of 0.5,
    # distributes it without fetching to device, and defines a compiled function that runs the embedding
    # on each replica. Takes one batch, replaces the second feature's weight with nothing, and runs the
    # lookup once without weights and once with them, unpacking per-replica outputs. Expects the weighted
    # result to equal the unweighted result scaled by 0.5, 1.0 and 1.0 for the three features (the second
    # has no weight and the third table uses a mean combiner so uniform weights cancel). Asserts closeness.
    strategy, mid_level_api = self._create_strategy_and_mid_level('sgd')
    weight = 0.5
    if ragged:
      dataset = self._create_ragged_dataset(
          strategy, include_weights=True, weight=weight)
    else:
      dataset = self._create_sparse_dataset(
          strategy, include_weights=True, weight=weight)

    dataset_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def embedding_lookup(features, weights):
      # [FUNCTION DOC]
      # HELPER: compiled wrapper of the weights test. Inputs: features and weights (weights may be nothing).
      # Runs the inner step on every replica via the strategy and returns the per-replica activations.

      def step(features, weights):
        # [FUNCTION DOC]
        # HELPER: per-replica body of the lookup. Inputs: features, weights. Calls the embedding object with
        # them and returns its activations.
        return mid_level_api(features, weights)

      return strategy.run(step, args=(features, weights))

    features, weights = next(dataset_iter)
    # Replace the weight for the second feature by None to test.
    weights = (weights[0], None, weights[2])

    no_weights_activations = embedding_lookup(features, weights=None)
    weights_activations = embedding_lookup(features, weights=weights)

    no_weights0 = (self._unpack(strategy, no_weights_activations[0]),
                   self._unpack(strategy, no_weights_activations[1]),
                   self._unpack(strategy, no_weights_activations[2]))
    weights0 = (self._unpack(strategy, weights_activations[0]),
                self._unpack(strategy, weights_activations[1]),
                self._unpack(strategy, weights_activations[2]))
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

if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
