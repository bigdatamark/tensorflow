# [MODULE DOC]
# TPU test module for the mid level TPU embedding API (version 2) under a model-parallel TPU
# strategy (one replica spread over two cores with SPMD XLA partitioning). Defines one test class
# that overrides strategy creation, provides an enqueue helper and trains a tiny one-table model
# with a decaying learning rate, comparing the learned table against analytically computed
# values. Runs under the TF test runner with V2 behavior when executed as a script. No public API.
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
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import test
from tensorflow.python.tpu import device_assignment as device_lib
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test


class TPUEmbeddingTPUStrategyV2Test(
    tpu_embedding_base_test.TPUEmbeddingBaseTest
):
  # [CLASS DOC]
  # Test case for the TPU embedding mid level API on a model-parallel TPU strategy: a single
  # replica using two cores per replica. Holds the replica and core counts as attributes, creates
  # the strategy itself (storing it and the flat list of embedding devices on the instance) and
  # reuses the other fixtures of the shared base test class.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: runs the base class setup, then records one replica and two cores per
    # replica as instance attributes used by the strategy creation and the expected values.
    super().setUp()

    self._num_replicas = 1
    self._num_cores_per_replica = 2

  def _get_strategy(self) -> tpu_strategy.TPUStrategy:
    # [FUNCTION DOC]
    # HELPER: overrides the base class strategy factory so the tests use a model-parallel TPU.
    # Initializes the TPU system (via the base helper), builds a device assignment with a
    # computation shape of 1 by 1 by 1 by 2 and one replica, and creates a TPU strategy (v2)
    # using the resolver with that assignment and SPMD XLA partitioning. Side effects: stores
    # the strategy and a flattened list of the TPU devices of all replicas on the instance.
    # Returns the strategy.
    topology = self._init_tpu_system()

    d_assign = device_lib.device_assignment(
        topology,
        computation_shape=[1, 1, 1, 2],
        num_replicas=1,
    )

    self.strategy = tpu_strategy.TPUStrategyV2(
        self.resolver,
        experimental_device_assignment=d_assign,
        experimental_spmd_xla_partitioning=True,
    )

    self.embedding_devices = sum(
        (list(replica) for replica in self.strategy.extended._tpu_devices), []
    )

    return self.strategy

  def enqueue(self, inp, mid_level_api, use_device, training):
    # [FUNCTION DOC]
    # HELPER: enqueues input batches into the embedding for the model-parallel tests.
    # Inputs: inp is a list of per-device feature batches; mid_level_api is the embedding
    # object; use_device is a flag choosing per-device enqueue; training is the training flag
    # forwarded to enqueue. When use_device is true each batch is enqueued onto its matching
    # embedding device; otherwise only the first batch is enqueued. Returns nothing.
    if use_device:
      for emb, device in zip(inp, self.embedding_devices):
        mid_level_api.enqueue(emb, device=device, training=training)
    else:
      mid_level_api.enqueue(inp[0], training=training)

  @parameterized.parameters(False, True)
  def test_spmd_training(self, use_device):
    # [FUNCTION DOC]
    # Parameterized test over use_device (false, true). Builds a one-table embedding (two rows,
    # width four, zero initial values, sum combiner) with an SGD optimizer whose learning rate
    # decays linearly from 1.0 to 0.5 over ten steps driven by a step counter variable. Inputs
    # are either one distributed dataset or one dataset per embedding device. A compiled
    # function enqueues, dequeues, takes gradients of the summed activations and applies them.
    # Runs ten steps, incrementing the counter each time, unpacks the activations per step and
    # asserts they match golden values: the negative running sums of the learning rates, repeated
    # across the table width and the cores per replica.
    num_steps = 10
    num_steps_float = float(num_steps)
    starting_lr = 1.0
    ending_lr = 0.5

    strategy = self._get_strategy()

    # Create model with Keras.
    with strategy.scope():
      step_counter = tf_variables.Variable(0.0, dtypes.float32)

      def lr_function():
        # [FUNCTION DOC]
        # Learning rate schedule callback: returns the larger of the ending rate and a linear
        # interpolation from the starting rate toward the ending rate based on the current value
        # of the step counter divided by the number of steps. Takes no arguments.
        return gen_math_ops.maximum(
            ending_lr,
            starting_lr
            + ((ending_lr - starting_lr) * step_counter) / num_steps_float,
        )

      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=lr_function)
      table_config = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=2,
          dim=4,
          initializer=init_ops_v2.Constant(np.zeros((2, 4))),
          combiner='sum',
          name='table',
      )
      mid_level_api = tpu_embedding_v2.TPUEmbedding(
          feature_config={
              'feature': tpu_embedding_v2_utils.FeatureConfig(
                  table=table_config, name='feature'
              )
          },
          optimizer=optimizer,
      )

    def input_fn(ctx):
      # [FUNCTION DOC]
      # Dataset factory for the distribution strategy: ignores the input context and returns an
      # endlessly repeating dataset whose single element is a mapping with one feature holding
      # the ids zero and one as a two by one integer tensor.
      del ctx
      feature = {
          'feature': constant_op.constant(
              [0, 1], shape=(2, 1), dtype=dtypes.int32
          )
      }
      return dataset_ops.DatasetV2.from_tensors(feature).repeat()

    def create_datasets():
      # [FUNCTION DOC]
      # Builds the input datasets for the enclosing test. If per-device input is requested, makes
      # one endlessly repeating dataset per embedding device, each holding a feature id equal to
      # the device index modulo the cores per replica. Otherwise distributes the input function
      # dataset through the strategy without fetching to device. Returns a list of datasets.
      """Creates either a per-replica dataset, or multiple per-devices ones.

      This function explicitly creates per-device datasets because the strategy
      does not produce a distributed dataset in the model-parallel case; there
      is only one replica. Without this consideration, the embeddings would be
      read as [0, 0] instead of the expected [0, 1] since all the devices would
      receive the same value.

      Returns:
        A list of one or more dataset(s).
      """
      if use_device:
        datasets = []
        for i in range(len(self.embedding_devices)):
          datasets.append(
              dataset_ops.DatasetV2.from_tensor_slices(
                  {'feature': [[[i % self._num_cores_per_replica]]]}
              ).repeat()
          )
        return datasets
      else:
        dataset = strategy.distribute_datasets_from_function(
            input_fn,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False
            ),
        )
        return [dataset]

    datasets = create_datasets()
    iterators = [iter(ds) for ds in datasets]

    @def_function.function(jit_compile=True)
    def test_fn():
      # [FUNCTION DOC]
      # Compiled (XLA jit) training step: takes the next element of every iterator, enqueues them
      # as training input through the enqueue helper, and runs the per-replica step on the
      # strategy, returning its activations.
      def step():
        # [FUNCTION DOC]
        # Per-replica step: dequeues activations, records them on a gradient tape, takes the
        # summed feature activations divided by the replica count as loss, computes gradients with
        # respect to the activations, applies them to the embedding tables and returns the
        # activations.
        with backprop.GradientTape() as tape:
          activations = mid_level_api.dequeue()
          tape.watch(activations)
          result = math_ops.reduce_sum(activations['feature'])
          loss = result / self._num_replicas
        grads = tape.gradient(loss, activations)
        mid_level_api.apply_gradients(grads)
        return activations

      inp = [next(it) for it in iterators]
      self.enqueue(inp, mid_level_api, use_device, training=True)
      return strategy.run(step)

    # Run model.
    results = []
    for _ in range(num_steps):
      result = test_fn()
      results.append(self._unpack(strategy, result['feature']))
      step_counter.assign_add(1.0)

    # Table is 2 elements wide, per-replica batch size of 1, with id 0.
    # Loss for the gradient is the sum of the entries divided by the number of
    # replicas. Thus the per replica gradient is 1/#of replicas for row 0 and no
    # other updates. The reduced gradient is therefore 1.
    # Learning rate schedule over num_steps steps:
    # 1.0 0.95 0.9 0.85 0.8 ...
    # Since use SGD and the gradient is one, the first row of the table is
    # [0, 0] [-1.0, -1.0] [-1.95, -1.95] [-2.85, -2.85] ... (the negative
    # partial sums of the above).

    learning_rates = [starting_lr - (starting_lr - ending_lr) / num_steps * j
                      for j in range(num_steps)]
    cumsum = [sum(learning_rates[0:j]) for j in range(num_steps)]
    goldens = [[[-cumsum[i]] * table_config.dim] * self._num_cores_per_replica
               for i in range(10)]
    self.assertAllClose(results, goldens)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
