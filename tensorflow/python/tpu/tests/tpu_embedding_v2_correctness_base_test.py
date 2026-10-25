# [MODULE DOC]
# HELPER: shared base for the V2 TPU embedding correctness test modules (sparse, ragged, hd variants).
# Defines a test case class holding the common routine that trains or evaluates a TPU embedding on a
# dataset and compares results with expected values. No tests of its own. Runs with v2 behavior via
# the test main entry. Builds on the shared TPU embedding base test module.
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
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.platform import test
from tensorflow.python.tpu.tests import tpu_embedding_base_test


class TPUEmbeddingCorrectnessBaseTest(
    tpu_embedding_base_test.TPUEmbeddingBaseTest):
  # [CLASS DOC]
  # HELPER: base class for the correctness test cases of the V2 mid-level API.
  # Adds one routine, the embedding test driver, used by subclasses with different input types and
  # modes. Inherits fixtures and checking helpers from the shared base test.

  def _test_embedding(self, optimizer_name, training, sparse,
                      is_high_dimensional):
    # [FUNCTION DOC]
    # HELPER: end-to-end driver used by subclass tests.
    # Inputs: optimizer_name, string naming the optimizer; training, whether to apply gradients; sparse,
    # whether to use sparse (otherwise ragged) inputs; is_high_dimensional, whether inputs have an extra
    # dimension. Creates strategy, embedding and optimizer, makes the dataset, builds the embedding with
    # explicit shapes for the high-dimensional case, distributes the dataset without fetching to device,
    # and defines a compiled function that enqueues a batch and runs a step per replica: dequeue
    # activations, compute total loss, and when training take gradients of per-replica loss and apply
    # them. Runs it, retrieves variables from TPU to CPU, gets the global sparse input and calls the
    # result checker. Returns nothing; failures surface as assertions.
    # KNOWN ISSUE: for non-sparse high-dimensional input the code still creates a high-dimensional
    # sparse dataset instead of a ragged one, so the ragged high-dimensional path is not really exercised
    # by this driver.
    strategy, mid_level_api, optimizer = (
        self._create_strategy_and_mid_level(optimizer_name))

    if sparse:
      if is_high_dimensional:
        dataset = self._create_high_dimensional_sparse_dataset(strategy)
      else:
        dataset = self._create_sparse_dataset(strategy)
    else:
      if is_high_dimensional:
        dataset = self._create_high_dimensional_sparse_dataset(strategy)
      else:
        dataset = self._create_ragged_dataset(strategy)

    if is_high_dimensional:
      if sparse:
        mid_level_api.build([
            TensorShape([self.batch_size, self.data_batch_size, 2]),
            TensorShape([self.batch_size, self.data_batch_size, 2]),
            TensorShape([self.batch_size, self.data_batch_size, 3]),
        ])
      else:
        mid_level_api.build([
            TensorShape([self.batch_size, self.data_batch_size, None]),
            TensorShape([self.batch_size, self.data_batch_size, None]),
            TensorShape([self.batch_size, self.data_batch_size, None]),
        ])

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False))
    dist_iter = iter(dist)

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # HELPER: compiled function of the driver. Takes nothing. Enqueues the next distributed batch with the
      # training flag, runs the step on each replica via the strategy, and returns the per-replica results.

      def step():
        # [FUNCTION DOC]
        # HELPER: per-replica computation of the driver. Returns a list of the total loss followed by the
        # activations. Without training it only dequeues, computes the loss and returns. With training it
        # dequeues inside a gradient recording context, watches the activations, computes the total loss and
        # a per-replica loss, takes gradients and applies them to the embedding.
        """Create and run computation that returns the embedding activations."""
        if not training:
          activations = mid_level_api.dequeue()
          total_loss = self._get_total_loss_tensor(activations)
          ret_val = [total_loss] + list(activations)
          return ret_val
        else:
          with backprop.GradientTape() as tape:
            activations = mid_level_api.dequeue()
            tape.watch(activations)
            total_loss = self._get_total_loss_tensor(activations)
            loss_per_replica = total_loss / strategy.num_replicas_in_sync
          gradients = tape.gradient(loss_per_replica, activations)
          mid_level_api.apply_gradients(gradients)
        ret_val = [total_loss] + list(activations)
        return ret_val

      mid_level_api.enqueue(next(dist_iter), training=training)
      result = strategy.run(step)
      return result

    # Run model.
    shard_out_val = test_fn()

    # Retrieve TPU weights to CPU.
    mid_level_api._retrieve_variables()

    # Compute sparse tensors for global batch.
    if is_high_dimensional:
      input_data = next(
          iter(self._create_high_dimensional_sparse_dataset(strategy)))
    else:
      input_data = next(iter(self._create_sparse_dataset(strategy)))

    # Check results.
    self._check_results(strategy, shard_out_val, training, input_data,
                        mid_level_api._variables, optimizer,
                        is_high_dimensional)

if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
