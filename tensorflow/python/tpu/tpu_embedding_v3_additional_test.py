# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Additional single and multi worker tests for the TPU embedding V3 layer. Defines a command
# line flag naming the TPU, two small helpers (per-replica value extraction and an array
# padding initializer) and one test class covering lookup, stacking and checkpoint restore
# between TPU and serving embeddings. Needs a TPU; running as a script enables v2 behavior and
# the MLIR bridge.
"""Additional multi/single worker tests for tpu_embedding_v3."""

import os

from absl import flags
from absl.testing import parameterized
import numpy as np

from tensorflow.python.checkpoint import checkpoint as tf_checkpoint
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework.constant_op import constant as tf_constant
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import save as tf_save
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tpu_embedding_for_serving
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_embedding_v3
from tensorflow.python.tpu import tpu_embedding_v3_utils


_TPU = flags.DEFINE_string('tpu', None, 'The TPU to use for TPUStrategy.')
# pylint: disable=g-long-lambda


RowIdInitializer = tpu_embedding_v2_utils.RowIdInitializer
shard_initializer = tpu_embedding_v3_utils.shard_initializer


def get_replica_values(per_replica_or_tensor):
  # [FUNCTION DOC]
  # HELPER: normalizes strategy outputs for the assertions in this module.
  # Input: a per-replica distributed value or a plain tensor. Returns the tuple of per-replica
  # values for a distributed value, otherwise a one-element list holding the input.
  if isinstance(per_replica_or_tensor, values_lib.PerReplica):
    return per_replica_or_tensor.values
  else:
    return [per_replica_or_tensor]


def pad_to_shape_initializer(init_mat):
  # [FUNCTION DOC]
  # HELPER: builds table initializers from explicit arrays for the tests in this module.
  # Input: init_mat, a two-dimensional tensor of initial values. Returns an initializer
  # callable taking a shape and a dtype (dtype ignored) that pads init_mat with constant
  # padding at the end of rows and columns up to the requested shape. Padding amounts become
  # negative (an error at call time) if the requested shape is smaller than init_mat.
  """An initializer that pads init_mat out to the given shape."""
  return lambda shape, dtype: array_ops.pad(
      init_mat,
      [
          [0, shape[0] - init_mat.shape[0]],
          [0, shape[1] - init_mat.shape[1]],
      ],
      'CONSTANT',
  )


class TPUEmbeddingLayerV2Test(parameterized.TestCase, test.TestCase):
  # [CLASS DOC]
  # Test case for the TPU embedding V3 layer on a real TPU (single or multi worker): initial
  # values, table stacking, lookup, and restoring checkpoints between the TPU layer and the
  # serving (CPU) embedding. Fixtures built in setUp: vocabulary size and dimension, a TPU
  # strategy with an explicit device assignment, sharded initializers, two table configs
  # (video, user) and their feature configs.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture run before each test. Connects to the TPU named by the flag (connecting to the
    # cluster when no flag is given), initializes the TPU system, builds a device assignment
    # from the topology and one replica per core, and creates a TPU strategy; registers a
    # cleanup that shuts the TPU system down. Defines sharded initializer helpers, the video
    # and user table and feature configs, and asserts the strategy's device grid has one column
    # and one row per core. Side effects: TPU system initialization and shutdown. Returns
    # nothing.
    super().setUp()
    self.vocabulary_size = 128
    self.embedding_dim = 8

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu=_TPU.value)
    if _TPU.value is None:
      remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)

    # FIXME(b/303466959): Remove this device assignment after TPUStrategy
    # can follow the actual device ordering under SC.
    topology = tpu_cluster_resolver.initialize_tpu_system(resolver)
    tpu_metadata = resolver.get_tpu_system_metadata()

    device_assignment = device_assignment_lib.DeviceAssignment.build(
        topology, num_replicas=tpu_metadata.num_cores
    )
    self._strategy = tpu_strategy.TPUStrategyV2(
        resolver, experimental_device_assignment=device_assignment
    )

    self.addCleanup(tpu_cluster_resolver.shutdown_tpu_system, resolver)
    self.sharded_row_id_initializer = lambda offset: shard_initializer(
        self._strategy, RowIdInitializer(offset)
    )

    self.sharded_pad_to_shape_initializer = lambda init_mat: shard_initializer(
        self._strategy, pad_to_shape_initializer(init_mat)
    )

    self.table_video = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=self.embedding_dim,
        initializer=self.sharded_row_id_initializer(0),
        combiner='sum',
        name='video',
    )
    self.table_user = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=self.embedding_dim,
        initializer=self.sharded_row_id_initializer(1000),
        combiner='sum',
        name='user',
    )
    self.feature_video = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_video,
        name='video',
        output_shape=[self.vocabulary_size],
    )
    self.feature_user = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_user, name='user', output_shape=[self.vocabulary_size]
    )

    self.assertEqual(
        self._strategy.extended._tpu_devices.shape, (tpu_metadata.num_cores, 1)
    )

  def testSingleTableInitializeAndLookup(self):
    # [FUNCTION DOC]
    # Scenario: one table (video) with a row-id initializer; builds the TPU embedding layer
    # inside the strategy scope with pipelining enabled and runs one compiled step on inputs
    # that look up every row once. Asserts each replica's returned embeddings are close to the
    # initializer's expected values for the whole table (rows equal to their row ids).
    # This test sets up devices to lookup the entire table.

    feature_config = [self.feature_video]

    strategy = self._strategy

    with strategy.scope():
      embedding_layer = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config,
          tpu_embedding_v2_utils.Adagrad(),
          pipeline_execution_with_tensor_core=True,
      )

      @def_function.function
      def train_step(features):
        # [FUNCTION DOC]
        # Compiled step used by the single-table test. Input: features, distributed input
        # values. Runs the inner per-replica function on every replica through the strategy
        # and returns the per-replica results (the first output of the embedding layer).
        def train_step_fn(features):
          # [FUNCTION DOC]
          # Per-replica body: calls the embedding layer on the features and returns the first
          # element of its output (the looked-up embeddings of the first feature).
          return embedding_layer(features)[0]

        return strategy.run(train_step_fn, args=(features,))

    def value_fn(ctx):
      # [FUNCTION DOC]
      # Input producer for distributed values. Input: ctx, the replica context (ignored).
      # Returns a list with one sparse tensor that looks up every row id of the vocabulary
      # exactly once (one id per example), with dense shape of vocabulary size by one.
      del ctx  # unused
      return [
          sparse_tensor.SparseTensor(
              indices=[[i, 0] for i in range(0, self.vocabulary_size)],
              values=np.arange(0, self.vocabulary_size),
              dense_shape=[self.vocabulary_size, 1],
          ),
      ]

    features = strategy.experimental_distribute_values_from_function(value_fn)

    [embeddings] = train_step(features)

    expected = RowIdInitializer(0)(
        shape=(self.vocabulary_size, self.embedding_dim), dtype=dtypes.float32
    )
    for replica in get_replica_values(embeddings):
      self.assertAllClose(replica, expected)

  def testStackedTableInitializeAndLookup(self):
    # [FUNCTION DOC]
    # Scenario: two tables (video and user) with different row-id offsets that are stacked by
    # the layer. Runs one compiled step looking up every row of both. Asserts each replica's
    # embeddings for video and user are close to the expected row-id values for each table
    # (offsets 0 and 1000), showing stacking keeps each table's own initial values.
    # This test sets up devices to lookup the entire table.
    feature_config = [self.feature_video, self.feature_user]

    strategy = self._strategy

    def value_fn(ctx):
      # [FUNCTION DOC]
      # Input producer for distributed values. Input: ctx, the replica context (ignored).
      # Returns a list of two identical sparse tensors, one per feature, each looking up every
      # row id of the vocabulary once, with dense shape of vocabulary size by one.
      del ctx  # unused
      return [
          sparse_tensor.SparseTensor(
              indices=[[i, 0] for i in range(0, self.vocabulary_size)],
              values=np.arange(0, self.vocabulary_size),
              dense_shape=[self.vocabulary_size, 1],
          ),
          sparse_tensor.SparseTensor(
              indices=[[i, 0] for i in range(0, self.vocabulary_size)],
              values=np.arange(0, self.vocabulary_size),
              dense_shape=[self.vocabulary_size, 1],
          ),
      ]

    features = strategy.experimental_distribute_values_from_function(value_fn)

    with strategy.scope():
      embedding_layer = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config,
          tpu_embedding_v2_utils.Adagrad(),
          pipeline_execution_with_tensor_core=True,
      )

      @def_function.function
      def train_step(features):
        # [FUNCTION DOC]
        # Compiled step used by the stacked-table test. Input: features, distributed input
        # values for two features. Runs the inner per-replica function on every replica through
        # the strategy and returns its per-replica results (the first output of the layer).
        def train_step_fn(features):
          # [FUNCTION DOC]
          # Per-replica body: calls the embedding layer on the features and returns the first
          # element of its output (the looked-up embeddings of the first feature).
          return embedding_layer(features)[0]

        return strategy.run(train_step_fn, args=(features,))

    [embeddings_video, embeddings_user] = train_step(features)

    expected_video = RowIdInitializer(0)(
        shape=(self.vocabulary_size, self.embedding_dim), dtype=dtypes.float32
    )
    expected_user = RowIdInitializer(1000)(
        shape=(self.vocabulary_size, self.embedding_dim), dtype=dtypes.float32
    )
    for replica_video, replica_user in zip(
        get_replica_values(embeddings_video),
        get_replica_values(embeddings_user),
    ):
      self.assertAllClose(replica_video, expected_video)
      self.assertAllClose(replica_user, expected_user)

  def testTwoTablesStackedHaveCorrectInitialValues(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name promises a check of initial values, but the test only asserts the
    # number of embedding tables; the initial values are never read or compared.
    # Scenario: two small tables with the same width and explicit initial arrays, padded by the
    # sharded initializer. Builds the layer in the strategy scope and asserts the two tables
    # were stacked into a single embedding table.
    table1_initial_value = np.arange(
        start=100, stop=120, dtype=np.float32
    ).reshape([10, 2])
    table1 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=10,
        dim=2,
        initializer=self.sharded_pad_to_shape_initializer(table1_initial_value),
        combiner='sum',
        name='table1',
    )
    table2_initial_value = np.arange(
        start=200, stop=240, dtype=np.float32
    ).reshape([20, 2])
    table2 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=20,
        dim=2,
        initializer=self.sharded_pad_to_shape_initializer(table2_initial_value),
        combiner='sum',
        name='table2',
    )
    feature_configs = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=table1, name='feature1', output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=table2, name='feature2', output_shape=[16]
        ),
    ]

    with self._strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
          pipeline_execution_with_tensor_core=True,
      )
      # The two tables should be stacked into the same variable.
      self.assertLen(mid_level_api.embedding_tables, 1)

  def testCpuRestoreForNoStackedTables(self):
    # [FUNCTION DOC]
    # Scenario: two tables with different widths (so they are not stacked) are saved from the
    # TPU embedding layer to a checkpoint in a temporary directory. A serving (CPU) embedding
    # with the same configuration restores it and is built. Asserts the layer had two tables,
    # the serving embedding has two tables, and looking up row 1 in each table returns the
    # expected saved initial values (102 and 103 for the first; 209 to 217 for the second).
    table1_initial_value = np.arange(
        start=100, stop=120, dtype=np.float32
    ).reshape([10, 2])
    table1 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=12,
        dim=2,
        initializer=self.sharded_pad_to_shape_initializer(table1_initial_value),
        combiner='sum',
        name='table1',
    )
    table2_initial_value = np.arange(
        start=200, stop=380, dtype=np.float32
    ).reshape([20, 9])
    table2 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=20,
        dim=9,  # to ensure stacking does not occur
        initializer=self.sharded_pad_to_shape_initializer(table2_initial_value),
        combiner='sum',
        name='table2',
    )
    feature_configs = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=table1, name='feature1', output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=table2, name='feature2', output_shape=[16]
        ),
    ]

    with self._strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )
      # The two tables should be *not* be stacked.
      self.assertLen(mid_level_api.embedding_tables, 2)
      # Save v3 embedding
      checkpoint = tf_checkpoint.Checkpoint(mid_level_api)
      checkpoint_prefix = os.path.join(self.create_tempdir().full_path, 'ckpt')
      checkpoint_path = checkpoint.save(checkpoint_prefix)

    # Restore in serving embedding
    with distribute_lib.get_strategy().scope():
      serving_embedding = tpu_embedding_for_serving.TPUEmbeddingForServing(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )
      checkpoint_for_restore = tf_checkpoint.Checkpoint(serving_embedding)
      checkpoint_for_restore.restore(checkpoint_path)
      serving_embedding.build()
    # Check that 2 tables exist in serving.
    self.assertLen(serving_embedding.embedding_tables, 2)
    look_for_row_idx_1 = [
        sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]
        ),
        sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]
        ),
    ]
    row_lookup = serving_embedding(look_for_row_idx_1)
    self.assertAllEqual(
        row_lookup[0],
        tf_constant(
            [
                102.0,
                103.0,
            ],
            shape=(1, 2),
        ),
    )
    self.assertAllEqual(
        row_lookup[1],
        tf_constant(
            [209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0],
            shape=(1, 9),
        ),
    )

  def testCpuRestoreForStackedTables(self):
    # [FUNCTION DOC]
    # Scenario: two tables of equal width are stacked into one variable in the TPU embedding
    # layer and saved to a checkpoint in a temporary directory; a serving (CPU) embedding with
    # the same configuration restores it. Asserts the layer had one table, the serving
    # embedding has two (unstacked on restore), lookups of row 1 give the expected saved values
    # for each table, then saves the serving embedding as a saved model, loads it back and
    # asserts the loaded object has two variables.
    table1_initial_value = np.arange(
        start=100, stop=120, dtype=np.float32
    ).reshape([10, 2])
    table1 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=12,
        dim=2,
        initializer=self.sharded_pad_to_shape_initializer(table1_initial_value),
        combiner='sum',
        name='table1',
    )
    table2_initial_value = np.arange(
        start=200, stop=240, dtype=np.float32
    ).reshape([20, 2])
    table2 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=20,
        dim=2,
        initializer=self.sharded_pad_to_shape_initializer(table2_initial_value),
        combiner='sum',
        name='table2',
    )
    feature_configs = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=table1, name='feature1', output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=table2, name='feature2', output_shape=[16]
        ),
    ]

    with self._strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )
      # The two tables should be stacked into the same variable.
      self.assertLen(mid_level_api.embedding_tables, 1)
      # Save v3 embedding
      checkpoint = tf_checkpoint.Checkpoint(mid_level_api)
      checkpoint_prefix = os.path.join(self.create_tempdir().full_path, 'ckpt')
      checkpoint_path = checkpoint.save(checkpoint_prefix)

    # Restore in serving embedding
    with distribute_lib.get_strategy().scope():
      serving_embedding = tpu_embedding_for_serving.TPUEmbeddingForServing(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )
      checkpoint_for_restore = tf_checkpoint.Checkpoint(serving_embedding)
      checkpoint_for_restore.restore(checkpoint_path)
      serving_embedding.build()
    # Check that unstacking happens on restore
    self.assertLen(serving_embedding.embedding_tables, 2)
    look_for_row_idx_1 = [
        sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]
        ),
        sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]
        ),
    ]
    row_lookup = serving_embedding(look_for_row_idx_1)
    self.assertAllEqual(
        row_lookup[0],
        tf_constant([102.0, 103.0], shape=(1, 2)),
    )
    self.assertAllEqual(
        row_lookup[1],
        tf_constant([202.0, 203.0], shape=(1, 2)),
    )
    saved_model_path = os.path.join(
        self.create_tempdir().full_path, 'saved_model'
    )
    tf_save.save(
        serving_embedding, saved_model_path
    )
    loaded_embedding = tf_load.load(saved_model_path)
    self.assertLen(loaded_embedding._variables, 2)

  def testUnshardedToTpuRestore(self):
    # [FUNCTION DOC]
    # Scenario: one table is created in a serving (CPU, unsharded) embedding and saved to a
    # checkpoint; a TPU embedding layer with the same configuration restores it and is built.
    # The test then computes the padded vocabulary size and, for every device in order and each
    # SparseCore on it, slices the unsharded value with a stride equal to the total shard
    # count, pads the slice to the shard shape, and asserts the value read from that device
    # equals the concatenation of the expected padded slices.
    table1 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=25,
        dim=6,
        initializer=self.sharded_row_id_initializer(0),
        combiner='sum',
        name='table1',
    )
    feature_configs = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=table1, name='feature1', output_shape=[16]
        ),
    ]

    with distribute_lib.get_strategy().scope():
      cpu_embedding = tpu_embedding_for_serving.TPUEmbeddingForServing(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.Adagrad(0.1),
      )

      self.assertLen(cpu_embedding.embedding_tables, 1)
      # Save unsharded embedding
      checkpoint = tf_checkpoint.Checkpoint(cpu_embedding)
      checkpoint_prefix = os.path.join(self.create_tempdir().full_path, 'ckpt')
      checkpoint_path = checkpoint.save(checkpoint_prefix)

    # Restore in TPU embedding
    strategy = self._strategy
    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.Adagrad(0.1),
      )
      checkpoint_for_restore = tf_checkpoint.Checkpoint(mid_level_api)
      checkpoint_for_restore.restore(checkpoint_path)
      mid_level_api.build()

    replicas, cores_per_replica = strategy.extended._tpu_devices.shape
    total_sc_shards = (
        replicas * cores_per_replica * mid_level_api._num_sc_per_chip
    )
    def get_padded_vocab_size(total_sc_shards, vocab_size):
      # [FUNCTION DOC]
      # HELPER: computes the expected padded row count for the restore test.
      # Inputs: total_sc_shards, number of SparseCore shards; vocab_size, row count. Returns
      # the smallest positive multiple of eight times the shard count that is at least the
      # vocabulary size (zero if the vocabulary size is not positive).
      # multiple of (8 * total_sc_shards) which is greater than or equal to
      # vocab size
      result = 0
      while result < vocab_size:
        result += 8 * total_sc_shards
      return result

    padded_vocab = get_padded_vocab_size(
        total_sc_shards, table1.vocabulary_size
    )
    unsharded_full_value = cpu_embedding._variables['table1']['parameters']
    shard_shape = [padded_vocab // total_sc_shards, 8]
    offset = 0
    ordered_devices = []
    for devices in strategy.extended._tpu_devices:  # pylint: disable=protected-access
      ordered_devices.extend(devices)
    for device in ordered_devices:
      partition = []
      for _ in range(mid_level_api._num_sc_per_chip):
        sh = unsharded_full_value[offset::total_sc_shards, :]
        padded_sh = pad_to_shape_initializer(sh)(shard_shape, dtypes.float32)
        partition.append(padded_sh)
        offset += 1
      # Check value at each partition
      self.assertAllEqual(
          mid_level_api._variables['table1']['parameters'].read_from_device(
              device
          ),
          array_ops.concat(partition, axis=0),
      )


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  config.enable_mlir_bridge()
  test.main()
