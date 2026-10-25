# [MODULE DOC]
# Test module for the enqueue modes of the V2 TPU embedding API: enqueue inside or outside TPU
# compilation, enqueue under nested control flow, non-direct inputs, and automatic training-mode
# detection. Needs a TPU; runs with v2 behavior via the test main entry. Uses the shared base module.
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
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.tpu.tests import tpu_embedding_base_test
from tensorflow.python.util import nest


class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):
  # [CLASS DOC]
  # Test case for enqueue behaviors, derived from the shared TPU embedding base test, using its
  # strategy, dataset and numpy helpers.

  @parameterized.parameters([True, False])
  def test_enqueue_with_outside_compilation(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over use_mlir. KNOWN ISSUE: when use_mlir is true it turns on the MLIR bridge through
    # global configuration and never turns it off, which can leak into later tests in the process.
    # Scenario: enqueue called inside the per-replica function (outside compilation) must match the
    # usual enqueue outside it. Builds the embedding with explicit input shapes, takes one sparse batch,
    # runs both compiled variants without training, extracts core 0 numpy outputs, asserts they match.
    if use_mlir:
      config.enable_mlir_bridge()

    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    mid_level_api.build([
        TensorShape((self.batch_size, 2)),
        TensorShape((self.batch_size, 2)),
        TensorShape((self.batch_size, 3))
    ])
    dataset = self._create_sparse_dataset(strategy)
    dataset_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def enqueue_with_outside_compilation(data):
      # [FUNCTION DOC]
      # HELPER: compiled function. Input: data. Runs a per-replica function which enqueues the features
      # with training off inside the replica and returns the dequeued activations.
      def get_activations(features):
        # [FUNCTION DOC]
        # HELPER: per-replica body. Input: features. Enqueues them with training off, returns dequeue output.
        mid_level_api.enqueue(features, training=False)
        return mid_level_api.dequeue()
      return strategy.run(get_activations, args=(data,))

    @def_function.function
    def enqueue_without_outside_compilation(data):
      # [FUNCTION DOC]
      # HELPER: compiled function. Input: data. Enqueues outside the replica function (training off), then
      # runs a per-replica dequeue and returns activations.
      def get_activations():
        # [FUNCTION DOC]
        # HELPER: per-replica body without inputs; returns the result of a dequeue.
        return mid_level_api.dequeue()
      mid_level_api.enqueue(data, training=False)
      return strategy.run(get_activations)

    features = next(dataset_iter)

    activations_oc = enqueue_with_outside_compilation(features)
    activations = enqueue_without_outside_compilation(features)

    # Extact per core numpy arrays.
    activations_oc0 = self._get_replica_numpy(activations_oc, strategy, 0)
    activations0 = self._get_replica_numpy(activations, strategy, 0)

    self.assertAllClose(activations_oc0, activations0)

  @parameterized.parameters(True, False)
  def test_enqueue_with_outside_compilation_in_control_flow(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over use_mlir. KNOWN ISSUE: enabling the MLIR bridge is global and never undone.
    # Skipped when project or zone flag is set. Scenario: enqueue wrapped inside a separate compiled
    # function (not inlined into the caller) is a form of control flow. Calling it in a replica function
    # must raise a runtime error saying the graph does not match the one containing the replicate context.
    self.skip_if_oss()
    if use_mlir:
      config.enable_mlir_bridge()

    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    dataset = self._create_sparse_dataset(strategy)
    dataset_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    # This is one way to force the enqueue in some control flow. @tf.functions
    # aren't inlined in the calling tf.function. An alternative would be to
    # place the enqueue in a switch_v2 or something similar.
    @def_function.function
    def enqueue_fn(features):
      # [FUNCTION DOC]
      # HELPER: compiled function. Input: features. Enqueues with training off; returns nothing.
      mid_level_api.enqueue(features, training=False)

    @def_function.function
    def enqueue_with_outside_compilation():
      # [FUNCTION DOC]
      # HELPER: compiled function without inputs. Runs a per-replica function that calls the enqueue
      # function and dequeues, using the next distributed batch; returns the activations.
      def get_activations(features):
        # [FUNCTION DOC]
        # HELPER: per-replica body. Input: features. Calls the enqueue function, then returns a dequeue.
        enqueue_fn(features)
        return mid_level_api.dequeue()
      return strategy.run(get_activations, args=(next(dataset_iter),))

    with self.assertRaisesRegex(
        RuntimeError,
        'does not match graph which contains TPUReplicateContext'):
      enqueue_with_outside_compilation()

  def test_enqueue_with_outside_compilation_non_direct_input(self):
    # [FUNCTION DOC]
    # Scenario: enqueue inside the replica function with inputs that were transformed on the TPU. Builds
    # with explicit shapes, doubles every feature before enqueue so inputs are not direct dataset inputs,
    # and expects a value error complaining the input lacks the TPU input identity attribute.
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    mid_level_api.build([
        TensorShape((self.batch_size, 2)),
        TensorShape((self.batch_size, 2)),
        TensorShape((self.batch_size, 3))
    ])
    dataset = self._create_sparse_dataset(strategy)
    dataset_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def enqueue_with_outside_compilation():
      # [FUNCTION DOC]
      # HELPER: compiled function without inputs. Runs the replica function on the next distributed batch
      # and returns its activations.
      def get_activations(features):
        # [FUNCTION DOC]
        # HELPER: per-replica body. Input: features. Multiplies each by two (to break direct input), enqueues
        # with training off, returns a dequeue.
        # This inserts a mul operation on the TPU to trigger the direct input
        # error.
        features = (features[0]*2, features[1]*2, features[2]*2)
        mid_level_api.enqueue(features, training=False)
        return mid_level_api.dequeue()
      return strategy.run(get_activations, args=(next(dataset_iter),))

    with self.assertRaisesRegex(
        ValueError, 'which does not have the `_tpu_input_identity` attr'):
      enqueue_with_outside_compilation()

  def test_enqueue_with_outside_compilation_auto_mode(self):
    # [FUNCTION DOC]
    # Scenario: the embedding detects training or inference mode automatically from the ops sharing an
    # operation name. Builds with explicit shapes, then runs one compiled function that enqueues,
    # dequeues and applies an all-ones gradient under name call2, and another that enqueues and dequeues
    # under name call1 without gradients (training flag not set, left default). Runs the gradient one
    # first and the other after on the same batch, and compares core 0 results. Expected: the later
    # activations equal the earlier minus updates of 0.3 times replicas (scaled by lookup counts, 2x for
    # rows looked up twice) for the video features and 0.1-based values for the user feature. Asserts closeness.
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    mid_level_api.build([
        TensorShape((self.batch_size, 2)),
        TensorShape((self.batch_size, 2)),
        TensorShape((self.batch_size, 3))
    ])
    dataset = self._create_sparse_dataset(strategy)
    dataset_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def enqueue_with_no_gradient_apply(data):
      # [FUNCTION DOC]
      # HELPER: compiled function. Input: data. Runs a replica function and returns activations.
      def get_activations(features):
        # [FUNCTION DOC]
        # HELPER: per-replica body. Input: features. Enqueues and dequeues under the name call1 without
        # applying gradients or setting a training flag; returns activations.
        # Note the lack of setting training=False, so training defaults to true
        # here even though we don't have apply gradients.
        # We detect the correct mode based on which ops exist that share the
        # same 'name'.
        mid_level_api.enqueue(features, name='call1')
        return mid_level_api.dequeue(name='call1')
      return strategy.run(get_activations, args=(data,))

    @def_function.function
    def enqueue_with_gradient_apply(data):
      # [FUNCTION DOC]
      # HELPER: compiled function. Input: data. Runs a replica function and returns activations.
      def get_activations(features):
        # [FUNCTION DOC]
        # HELPER: per-replica body. Input: features. Enqueues and dequeues under the name call2, applies
        # gradients of all ones, and returns the activations obtained before the update.
        mid_level_api.enqueue(features, name='call2')
        activations = mid_level_api.dequeue(name='call2')
        # Apply an all ones gradient
        gradients = nest.map_structure(array_ops.ones_like, activations)
        mid_level_api.apply_gradients(gradients, name='call2')
        return activations
      return strategy.run(get_activations, args=(data,))

    data = next(dataset_iter)
    before_gradient_apply = enqueue_with_gradient_apply(data)
    after_gradient_apply = enqueue_with_no_gradient_apply(data)
    before_gradient_apply0 = self._get_replica_numpy(before_gradient_apply,
                                                     strategy, 0)
    after_gradient_apply0 = self._get_replica_numpy(after_gradient_apply,
                                                    strategy, 0)

    num_replicas = strategy.num_replicas_in_sync
    # We are passing a gradient of 1 for all lookups, optimizer is SGD with a
    # learning rate of 0.1. Feature 0 and 1 are looked up with a sum combiner
    # with the following ids:
    # Feature 0: [0, 0, 1], [0, 1, 1], ... repeated over num_replicas
    # Feature 1: [0, 1, 1], [0, 0, 1], ... repeated over num_replicas
    # i.e. Row 0 and 1 were looked up 3*num_replicas times over all cores and as
    # the gradient is 1, the accumulated gradient is 3*num_replicas for each
    # position in row 0 and 1 in table.
    #
    # See comments in test_pass_none_to_apply_gradients for the update to
    # Feature 2 and its table.
    # The *2 in the next tests are because those rows have 2 lookups vs
    # the 1 lookup in the other row.
    update = ([[0.3 * num_replicas], [0.3 * num_replicas * 2]],
              [[0.3 * num_replicas * 2], [0.3 * num_replicas]],
              [[0.1 * num_replicas], [0.1 / 3 * num_replicas]])
    golden = tuple([before - np.array(up) for before, up in
                    zip(before_gradient_apply0, update)])

    self.assertAllClose(golden, after_gradient_apply0)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
