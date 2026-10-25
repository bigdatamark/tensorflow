# [MODULE DOC]
# Hardware tests for the TPU embedding v3 (SparseCore) layer. Requires a real TPU cluster; each
# test connects to it, builds a TPU strategy and compares lookups, gradient updates, statistics
# computation, error handling and pipelining flags against expectations or the CPU serving
# implementation. Contains a helper that generates random hardware-compliant sparse input and one
# test class. Import time side effects: defines the shard initializer alias. Run as a script it
# enables v2 behavior and the MLIR bridge before starting the test runner.
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
import contextlib
import itertools
import logging

from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import test
from tensorflow.python.tpu import embedding_context_utils as ecu
from tensorflow.python.tpu import tpu_embedding_for_serving
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_embedding_v3
from tensorflow.python.tpu import tpu_embedding_v3_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.util import nest

shard_initializer = tpu_embedding_v3_utils.shard_initializer


def create_input_data_based_on_hw_requirement(
    num_chip,
    max_unique_ids_per_partition,
    per_sc_vocab_size,
    per_sc_sample_count,
    num_minibatches_per_physical_sparse_core,
):
  # [FUNCTION DOC]
  # HELPER: generates random coordinate-format input (row ids, column ids, gains) that respects
  # SparseCore hardware limits so that a given number of minibatches per core is required.
  # Inputs: chip count (4 sparse cores per chip are assumed); maximum unique ids per physical
  # replica; vocabulary rows per sparse core; samples per sparse core; desired minibatches per
  # physical sparse core.
  # Behavior: for every chip, core and physical replica it draws an id count between one more
  # than the capacity of all-but-one minibatches and the capacity of all minibatches, picks
  # unique local column ids per minibatch division without replacement, random row ids within the
  # core's samples and random gains, and maps them to global ids. Uses global numpy randomness.
  # Returns three numpy arrays: row ids, column ids and gains.
  # Raises an assertion error if a division of the vocabulary is smaller than the maximum unique
  # ids per partition (the minibatch count would not fit).
  # KNOWN ISSUE: a source comment says to make sure at least one replica contains something but
  # the sentence is unfinished; and the docstring's output description omits that splits are
  # not returned (only three arrays come back, not "gains and splits").
  """Create the coo tensor based on hardware requirements.

  Args:
    num_chip: number of chips in the tpu system.
    max_unique_ids_per_partition: max unique ids per physical replica
    per_sc_vocab_size: per sc shard of table size.
    per_sc_sample_count: per sc sample count.
    num_minibatches_per_physical_sparse_core: per sc minibatch number.

  Returns:
    row_ids, col_ids, gains and splits
  """
  num_sc_per_chip = 4
  num_physical_replica = num_chip * num_sc_per_chip

  col_ids = []
  row_ids = []
  gains = []

  smallest_num_division = np.power(
      2, np.ceil(np.log2(num_minibatches_per_physical_sparse_core))
  )
  division_size = (
      per_sc_vocab_size + smallest_num_division - 1
  ) // smallest_num_division

  assert division_size >= max_unique_ids_per_partition, (
      'The max_unique_ids_per_partition is set to'
      f' {max_unique_ids_per_partition} and the number of minibatches per'
      f' sparse core is set to {num_minibatches_per_physical_sparse_core}.'
      f' But the vocab size per sparse core is {per_sc_vocab_size} which is'
      ' not going to fit that many minibatches, consider setting the number of'
      ' minibatches smaller.'
  )

  # Generating id nums for each sc on a chip. Since each chip will have the
  # same number of ids, we can shuffle this array to get random id numbers for
  # each chip.
  # Make sure that at least 1 replica contains

  per_sc_per_minibatch_id_nums_for_each_replica = np.random.randint(
      max_unique_ids_per_partition
      * (num_minibatches_per_physical_sparse_core - 1)
      + 1,
      max_unique_ids_per_partition * num_minibatches_per_physical_sparse_core
      + 1,
      size=num_physical_replica,
  )

  per_chip_sample_count = per_sc_sample_count * num_sc_per_chip

  for chip_id in range(num_chip):
    for sc_id in range(num_sc_per_chip):
      np.random.shuffle(per_sc_per_minibatch_id_nums_for_each_replica)
      for physical_replica_id in range(num_physical_replica):
        physical_replica_id_nums = (
            per_sc_per_minibatch_id_nums_for_each_replica[physical_replica_id]
        )
        # Generate local col ids based on the minibatch constrains.
        # Make sure that the generated col ids are all unique.
        local_col_ids = np.array([])
        for i in range(num_minibatches_per_physical_sparse_core):
          local_col_ids_minibatch_size = max_unique_ids_per_partition
          if i == num_minibatches_per_physical_sparse_core - 1:
            local_col_ids_minibatch_size = (
                physical_replica_id_nums - i * max_unique_ids_per_partition
            )

          local_col_ids = np.append(
              local_col_ids,
              np.random.choice(
                  np.arange(division_size),
                  size=local_col_ids_minibatch_size,
                  replace=False,
              )
              + i * division_size,
          )
        local_row_ids = np.random.randint(
            low=0,
            high=per_sc_sample_count,
            size=physical_replica_id_nums,
        )

        row_ids += list(
            local_row_ids
            + chip_id * per_chip_sample_count
            + sc_id * per_sc_sample_count
        )
        col_ids += list(
            local_col_ids * num_physical_replica + physical_replica_id
        )

        gains += list(np.random.random(size=physical_replica_id_nums))

  return np.array(row_ids), np.array(col_ids), np.array(gains)


class TPUEmbeddingV3Test(parameterized.TestCase, test.TestCase):
  # [CLASS DOC]
  # Hardware tests of the TPU embedding v3 layer (SparseCore): lookups compared against the CPU
  # serving implementation, embedding initialization, backward passes and training, id
  # dropping, minibatching and checkpoint-related behavior. These tests need a real TPU
  # cluster. Shared fixtures (from setUp): two table configurations ("video" with constant 1.0 and
  # "user" with constant 2.0, vocabulary 16384, width 127, sum combiner) and a seeded random
  # generator.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture. Calls the base setUp, then creates the vocabulary size (16384), embedding
    # width (127), the two table configurations (video: constant 1.0, user: constant 2.0, both
    # with sum combiner) and a random generator seeded with zero. Returns nothing.
    super().setUp()
    self.vocabulary_size = 16384
    self.embedding_dim = 127
    self.table_video = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=self.embedding_dim,
        initializer=init_ops_v2.Constant(1.0),
        combiner='sum',
        name='video',
    )
    self.table_user = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=self.embedding_dim,
        initializer=init_ops_v2.Constant(2.0),
        combiner='sum',
        name='user',
    )
    self.rng = np.random.default_rng(0)

  def test_single_feature_single_table_lookup_with_static_buffer_size(self):
    # [FUNCTION DOC]
    # Scenario (needs a TPU): one feature on one table with output shape 16, fed a sparse batch
    # of 16 samples and 1024 ids, distributed across replicas. Runs the lookup on the TPU and on
    # the CPU serving implementation and asserts the activations are exactly equal.
    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_video, name='watched', output_shape=[16]
    )

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(1024)],
            values=np.arange(1024),
            dense_shape=[16, 1024],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Graph function that runs the lookup on every TPU replica with the distributed batch
      # captured from the enclosing test, and returns the per-replica results of the strategy run.
      def step(data):
        # [FUNCTION DOC]
        # Per-replica step: calls the embedding layer on the batch (one sparse feature) and
        # returns its activations.
        return mid_level_api(data)

      return strategy.run(step, args=(data,))

    result = test_fn()

    mid_level_api_cpu = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config,
        optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
    )

    cpu_result = mid_level_api_cpu(data)

    for per_feature_result, per_feature_result_cpu in zip(
        nest.flatten(result[0]), nest.flatten(cpu_result)
    ):
      self.assertAllEqual(per_feature_result, per_feature_result_cpu)

  def test_two_features_single_table_lookup_with_csr_input(self):
    # [FUNCTION DOC]
    # Scenario (needs a TPU): two features sharing one table, fed the same sparse batch of 16
    # samples. Runs the lookup on the TPU and on the CPU serving implementation with the same
    # table settings and asserts the activations of each feature are exactly equal.
    feature_config = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, output_shape=[16]
        ),
    ]

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(512)],
            values=np.arange(512),
            dense_shape=[16, 512],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Graph function that runs the two-feature lookup on every TPU replica with the distributed
      # batch captured from the enclosing test, and returns the per-replica results.
      def step(data):
        # [FUNCTION DOC]
        # Per-replica step: feeds the same batch as both features to the embedding layer and
        # returns its activations.
        return mid_level_api([data, data])

      return strategy.run(step, args=(data,))

    result = test_fn()

    mid_level_api_cpu = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config,
        optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
    )

    cpu_result = mid_level_api_cpu([data, data])

    for per_feature_result, per_feature_result_cpu in zip(
        nest.flatten(result[0]), nest.flatten(cpu_result[0])
    ):
      self.assertAllEqual(per_feature_result, per_feature_result_cpu)

  def test_two_features_two_tables_stacked_lookup_with_csr_input(self):
    # [FUNCTION DOC]
    # Scenario (needs a TPU): two features on two different tables with table stacking enabled
    # (64 max ids per chip per sample, no id dropping). Asserts the layer holds exactly one
    # (stacked) embedding table, then runs the same batch for both features on the TPU and on
    # the CPU serving implementation and asserts the activations are exactly equal.
    feature_config = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_user, output_shape=[16]
        ),
    ]

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(512)],
            values=np.arange(512),
            dense_shape=[16, 512],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    sparse_core_embedding_config = tpu_embedding_v3.SparseCoreEmbeddingConfig(
        disable_table_stacking=False,
        max_ids_per_chip_per_sample=64,
        allow_id_dropping=False,
    )

    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
          sparse_core_embedding_config=sparse_core_embedding_config,
      )
    self.assertLen(mid_level_api.embedding_tables, 1)

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Graph function that runs the two-feature lookup on every TPU replica with the distributed
      # batch captured from the enclosing test, and returns the per-replica results.
      def step(data):
        # [FUNCTION DOC]
        # Per-replica step: feeds the same batch as both features to the embedding layer (whose
        # two tables are stacked) and returns its activations.
        return mid_level_api([data, data])

      return strategy.run(step, args=(data,))

    result = test_fn()

    mid_level_api_cpu = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config,
        optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
    )

    cpu_result = mid_level_api_cpu([data, data])

    for per_feature_result, per_feature_result_cpu in zip(
        nest.flatten(result[0]), nest.flatten(cpu_result[0])
    ):
      self.assertAllEqual(per_feature_result, per_feature_result_cpu)

  def test_embedding_initialization(self):
    # [FUNCTION DOC]
    # Scenario (needs a TPU): three tables of width 128 with different initializers - constant
    # 1, a sharded initializer producing each element's own index, and a shard-aware constant
    # 23. Three features (one per table) are fed a batch of 16 ids with values offset by 0, 16
    # and 32. The batch is enqueued and dequeued through the embedding layer. Asserts each
    # output has shape 16 by 128 and the values: ones for the first, 2048 plus the element
    # index for the second (the ids start at row 16 of width 128), and 23 for the third.
    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    def element_id_initializer(shape, dtype):
      # [FUNCTION DOC]
      # HELPER: table initializer for the test. Given a two-dimensional shape and dtype, returns
      # a tensor in which each element holds its own row-major position (0, 1, 2, ...).
      values = math_ops.range(0, shape[0] * shape[1], dtype=dtype)
      return array_ops.reshape(values, shape)

    def sharding_aware_const_initializer(shape, dtype, shard_info=None):
      # [FUNCTION DOC]
      # HELPER: table initializer for the test that accepts optional shard info. Returns a
      # tensor filled with 23.0 whose shape is the shard's shape when shard info is given and
      # the full shape otherwise.
      initializer = init_ops_v2.Constant(23.0)
      return initializer(shard_info.shape if shard_info else shape, dtype=dtype)

    table_video = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=128,
        initializer=init_ops_v2.Constant(1.0),
        combiner='sum',
        name='video',
    )

    table_user = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=128,
        initializer=shard_initializer(strategy, element_id_initializer),
        combiner='sum',
        name='user',
    )

    table_country = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=128,
        initializer=sharding_aware_const_initializer,
        combiner='sum',
        name='country',
    )
    feature_config = {
        'watched': tpu_embedding_v2_utils.FeatureConfig(
            table=table_video, output_shape=[16]
        ),
        'user': tpu_embedding_v2_utils.FeatureConfig(
            table=table_user, output_shape=[16]
        ),
        'country': tpu_embedding_v2_utils.FeatureConfig(
            table=table_country, output_shape=[16]
        ),
    }

    sparse_features = {
        'watched': sparse_ops.sparse_reorder(
            sparse_tensor.SparseTensor(
                indices=[[i, 0] for i in range(16)],
                values=np.arange(16),
                dense_shape=[16, 1],
            )
        ),
        'user': sparse_ops.sparse_reorder(
            sparse_tensor.SparseTensor(
                indices=[[i, 0] for i in range(16)],
                values=np.arange(16) + 16,
                dense_shape=[16, 1],
            )
        ),
        'country': sparse_ops.sparse_reorder(
            sparse_tensor.SparseTensor(
                indices=[[i, 0] for i in range(16)],
                values=np.arange(16) + 32,
                dense_shape=[16, 1],
            )
        ),
    }

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )

    @def_function.function
    def test_fn(data):
      # [FUNCTION DOC]
      # Graph function taking a distributed batch: enqueues it on the first worker device to get
      # a partitioned tensor, then runs a per-replica dequeue and returns the activations
      # (a mapping from feature name to a per-replica tensor).
      def step(partitioned_tensor):
        # [FUNCTION DOC]
        # Per-replica step: dequeues the activations for the partitioned tensor, ignores the
        # second dequeue output and returns the activations.
        activations, _ = mid_level_api.dequeue(partitioned_tensor)
        return activations

      partitioned_tensor = mid_level_api.enqueue(
          data, device=strategy.extended.worker_devices[0]
      )
      return strategy.run(step, args=(partitioned_tensor,))

    result = test_fn(data)
    watched = result['watched']
    user = result['user']
    country = result['country']
    self.assertEqual(watched.shape, (16, 128))
    self.assertEqual(user.shape, (16, 128))
    self.assertEqual(country.shape, (16, 128))
    self.assertAllClose(
        watched, np.ones(16 * 128).reshape((16, 128)), atol=1e-5, rtol=1e-5
    )
    self.assertAllClose(
        user,
        2048 + np.arange(16 * 128).reshape((16, 128)),
        atol=1e-5,
        rtol=1e-5,
    )
    self.assertAllClose(
        country,
        np.ones(16 * 128).reshape((16, 128)) * 23.0,
        atol=1e-5,
        rtol=1e-5,
    )

  def _recover_same_sized_tables(self, table, strategy, num_tables=1):
    # [FUNCTION DOC]
    # HELPER: reads back table values from a sharded (possibly stacked) embedding variable.
    # KNOWN ISSUE: the loop variable named table shadows the table parameter (harmless since
    # the parameter is not used afterwards).
    # Inputs: table, a distributed variable whose values are per-device shards; strategy, the
    # TPU strategy (gives the replica count; 4 sparse cores per replica are assumed); num_tables
    # (default 1), the number of equally sized tables stacked in the variable.
    # Behavior: concatenates the shards (dropping padded columns beyond the embedding width),
    # undoes the mod sharding across all cores, splits the result into equal slices of the
    # vocabulary size, and undoes the per-table stacking rotation (four cores per table index).
    # Returns the single table if one table, otherwise a list of tables in stacking order.
    # This table has num_sparse_cores mod shards, so we need to slice,
    # reconcat and reshape.
    def _unshuffle_from_sc_to_cpu(num_sc_devices, t):
      # [FUNCTION DOC]
      # HELPER: undoes the mod sharding of a stacked table. Inputs: number of sparse core
      # devices and a two-dimensional tensor whose rows are laid out shard by shard. Returns a
      # tensor of the same shape with rows interleaved back into logical order. Raises an
      # assertion error if the row count is not a multiple of the device count.
      old_shape = t.shape
      # The width of the table must be a multiple of number of SC devices. The
      # tpu strategy does this round off at training time so we expect the
      # checkpoints value to meet this requirement.
      assert t.shape[0] % num_sc_devices == 0
      intermediate_tensor = array_ops.reshape(
          t, (num_sc_devices, t.shape[0] // num_sc_devices, t.shape[1])
      )
      intermediate_tensor = array_ops.transpose(intermediate_tensor, (1, 0, 2))
      return array_ops.reshape(intermediate_tensor, old_shape)

    table_partitions = [
        shard.numpy()[:, : self.embedding_dim] for shard in table.values
    ]
    full_table = np.concatenate(table_partitions, axis=0)
    full_table = _unshuffle_from_sc_to_cpu(
        strategy.num_replicas_in_sync * 4, full_table
    )

    # If we have multiple tables stacked, assume each has the same vocab sizez
    # and so was rounded the same (before stacking)
    slice_size = full_table.shape[0] // num_tables
    tables = []
    for i in range(num_tables):
      table = full_table[
          i * slice_size : i * slice_size + self.vocabulary_size, :
      ]
      # Since we apply the table stacking shift to the stacked table, we
      # are shifting it back here.
      table = np.reshape(
          table, (4 * strategy.num_replicas_in_sync, -1, table.shape[-1])
      )
      table = np.roll(table, -4 * i, axis=0)
      table = np.reshape(table, (-1, table.shape[-1]))
      tables.append(table)

    if num_tables == 1:
      return tables[0]

    return tables

  def test_single_feature_single_table_backwards_pass_with_csr_input(self):
    # [FUNCTION DOC]
    # Scenario (needs a TPU): one feature on one table (all values 1.0) with plain gradient
    # descent whose learning rate comes from a variable read function (value 1.0). A batch of
    # 16 samples and 1024 ids is enqueued and random gradients (16 by embedding width) are
    # applied. The table is recovered from the sharded variables and compared with a golden
    # table: ones, with each of the first 1024 rows reduced by the gradient of its sample (row
    # index modulo 16). Asserts the two are close.
    # Uses random gradients drawn with global numpy randomness.
    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_video, name='watched', output_shape=[16]
    )

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(1024)],
            values=np.arange(1024),
            dense_shape=[16, 1024],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    with strategy.scope():
      # Feed in learning rate as a variable.
      weight = tf_variables.Variable(initial_value=1.0)
      # Note that we return the function read_value and not its result here
      # that is important.
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=weight.read_value)
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config, optimizer=optimizer
      )
      mid_level_api.build()
    random = np.random.uniform(size=(16, self.embedding_dim)).astype(np.float32)

    @def_function.function
    def test_fn(random_input):
      # [FUNCTION DOC]
      # Graph function taking a gradient tensor (random_input); runs the step on every replica
      # with the distributed batch captured from the enclosing test. Returns nothing.
      def step(data, random_input):
        # [FUNCTION DOC]
        # Per-replica training step: enqueues the batch, copies the preprocessed tensors to the
        # device through outside compilation, then applies the given gradients to the embedding
        # tables (using the learning-rate variable). Returns nothing.
        partitioned_tensors = mid_level_api.enqueue(data)
        preprocessed_results = tpu_replication.outside_compilation(
            mid_level_api._copy_tensors_to_device,
            partitioned_tensors=partitioned_tensors,
        )
        mid_level_api.apply_gradients(random_input, preprocessed_results)

      strategy.run(step, args=(data, random_input))

    test_fn(random)

    full_table = self._recover_same_sized_tables(
        mid_level_api.embedding_tables['video'], strategy
    )

    golden = np.ones(
        [self.vocabulary_size, self.embedding_dim], dtype=np.float32
    )
    for i in range(1024):
      golden[i, :] = golden[i, :] - random[i % 16]

    self.assertAllClose(full_table, golden)

  def test_two_feature_single_table_backwards_pass_with_csr_input(self):
    # [FUNCTION DOC]
    # Scenario (needs a TPU): two features sharing the same table (all values 1.0), plain
    # gradient descent with learning rate 1.0 supplied through a variable read function. The same
    # batch of 16 samples and 512 ids is used for both features and a different random gradient
    # is applied to each. After recovering the table from the sharded variables, asserts it is
    # close to a golden table: ones with each of the first 512 rows reduced by both gradients of
    # its sample (row index modulo 16). Uses global numpy randomness.
    feature_config = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='watched', output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='watched2', output_shape=[16]
        ),
    ]

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(512)],
            values=np.arange(512),
            dense_shape=[16, 512],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    with strategy.scope():
      # Feed in learning rate as a variable.
      weight = tf_variables.Variable(initial_value=1.0)
      # Note that we return the function read_value and not its result here
      # that is important.
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=weight.read_value)
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config, optimizer=optimizer
      )
      mid_level_api.build()
    random1 = np.random.uniform(size=(16, self.embedding_dim)).astype(
        np.float32
    )
    random2 = np.random.uniform(size=(16, self.embedding_dim)).astype(
        np.float32
    )

    @def_function.function
    def test_fn(random_input):
      # [FUNCTION DOC]
      # Graph function taking a list of two gradient tensors (one per feature); runs the step on
      # every replica with the distributed batch captured from the enclosing test. Returns nothing.
      def step(data, random_input):
        # [FUNCTION DOC]
        # Per-replica training step: enqueues the batch as both features, copies the
        # preprocessed tensors to the device through outside compilation, then applies the given
        # gradients. Returns nothing.
        partitioned_tensors = mid_level_api.enqueue([data, data])
        preprocessed_results = tpu_replication.outside_compilation(
            mid_level_api._copy_tensors_to_device,
            partitioned_tensors=partitioned_tensors,
        )
        mid_level_api.apply_gradients(random_input, preprocessed_results)

      strategy.run(step, args=(data, random_input))

    test_fn([random1, random2])

    full_table = self._recover_same_sized_tables(
        mid_level_api.embedding_tables['video'], strategy
    )

    golden = np.ones(
        [self.vocabulary_size, self.embedding_dim], dtype=np.float32
    )
    for i in range(512):
      golden[i, :] = golden[i, :] - random1[i % 16] - random2[i % 16]

    self.assertAllClose(full_table, golden)

  def test_two_feature_two_tables_stacked_backwards_pass_with_csr_input(self):
    # [FUNCTION DOC]
    # Scenario (needs a TPU): two features on two different tables (video all 1.0, user all
    # 2.0) with table stacking enabled and plain gradient descent at learning rate 1.0. Random
    # gradients are applied per feature. The stacked table (named user_video) is recovered and
    # split into two tables; asserts both are close to golden tables where each of the first
    # 512 rows is reduced by the gradient of its sample (row modulo 16). Note that the user
    # table receives the second feature's gradient and the video table the first's.
    # Uses global numpy randomness.
    feature_config = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='watched', output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_user, name='watched2', output_shape=[16]
        ),
    ]

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(512)],
            values=np.arange(512),
            dense_shape=[16, 512],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    sparse_core_embedding_config = tpu_embedding_v3.SparseCoreEmbeddingConfig(
        disable_table_stacking=False,
        max_ids_per_chip_per_sample=64,
        allow_id_dropping=False,
    )

    with strategy.scope():
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=1.0)
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=optimizer,
          sparse_core_embedding_config=sparse_core_embedding_config,
      )
      mid_level_api.build()
    random1 = np.random.uniform(size=(16, self.embedding_dim)).astype(
        np.float32
    )
    random2 = np.random.uniform(size=(16, self.embedding_dim)).astype(
        np.float32
    )

    @def_function.function
    def test_fn(random_input):
      # [FUNCTION DOC]
      # Graph function taking a list of two gradient tensors (one per feature); runs the step on
      # every replica with the distributed batch captured from the enclosing test. Returns nothing.
      def step(data, random_input):
        # [FUNCTION DOC]
        # Per-replica training step: enqueues the batch as both features, copies the
        # preprocessed tensors to the device through outside compilation, then applies the given
        # gradients to the stacked tables. Returns nothing.
        partitioned_tensors = mid_level_api.enqueue([data, data])
        preprocessed_results = tpu_replication.outside_compilation(
            mid_level_api._copy_tensors_to_device,
            partitioned_tensors=partitioned_tensors,
        )
        mid_level_api.apply_gradients(random_input, preprocessed_results)

      strategy.run(step, args=(data, random_input))

    test_fn([random1, random2])

    full_tables = self._recover_same_sized_tables(
        mid_level_api.embedding_tables['user_video'], strategy, num_tables=2
    )

    goldens = [
        np.full(
            [self.vocabulary_size, self.embedding_dim], 2, dtype=np.float32
        ),
        np.ones([self.vocabulary_size, self.embedding_dim], dtype=np.float32),
    ]

    for i in range(512):
      goldens[0][i, :] = goldens[0][i, :] - random2[i % 16]
      goldens[1][i, :] = goldens[1][i, :] - random1[i % 16]

    self.assertAllClose(full_tables, goldens)

  def test_compute_sparse_core_stats_and_pass_it_to_api(self):
    # [FUNCTION DOC]
    # Scenario (needs a TPU): profiles a distributed batch with the layer's static stats
    # computation (max ids and max unique ids per table) for one feature, asserting the feature
    # configuration is not modified (table width stays 127). Feeds the stats to the layer through
    # the SparseCore configuration (stacking on, 64 max ids per chip per sample, no id dropping),
    # runs the lookup on the TPU, and asserts the activations equal those of the CPU serving
    # implementation. Also re-asserts the feature configuration is unchanged after the lookup.
    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_video, name='watched', output_shape=[16]
    )

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(1024)],
            values=np.arange(1024),
            dense_shape=[16, 1024],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    num_tpu_chips = strategy.num_replicas_in_sync

    # profile the dataset to get the max ids per table and max unique ids per
    # table.
    table_to_max_ids, table_to_max_unique_ids = (
        tpu_embedding_v3.TPUEmbeddingV2.compute_sparse_core_stats(
            features=data,
            feature_config=feature_config,
            num_tpu_chips=num_tpu_chips,
            optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
        )
    )
    self.assertEqual(
        feature_config.table.dim, 127, 'Unexpected update to FeatureConfig'
    )

    sparse_core_embedding_config = tpu_embedding_v3.SparseCoreEmbeddingConfig(
        disable_table_stacking=False,
        max_ids_per_chip_per_sample=64,
        max_ids_per_table=table_to_max_ids,
        max_unique_ids_per_table=table_to_max_unique_ids,
        allow_id_dropping=False,
    )

    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
          sparse_core_embedding_config=sparse_core_embedding_config,
      )

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Graph function that runs the lookup on every TPU replica with the distributed batch
      # captured from the enclosing test, and returns the per-replica results of the strategy run.
      def step(data):
        # [FUNCTION DOC]
        # Per-replica step: calls the embedding layer on the batch (one sparse feature) and
        # returns its activations.
        return mid_level_api(data)

      return strategy.run(step, args=(data,))

    result = test_fn()
    self.assertEqual(
        feature_config.table.dim, 127, 'Unexpected update to FeatureConfig'
    )
    mid_level_api_cpu = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config,
        optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
    )

    cpu_result = mid_level_api_cpu(data)

    for per_feature_result, per_feature_result_cpu in zip(
        nest.flatten(result[0]), nest.flatten(cpu_result)
    ):
      self.assertAllEqual(per_feature_result, per_feature_result_cpu)

  def test_compute_sparse_core_stats_shared_table(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test makes no assertion on the returned statistics; it only checks that
    # the call completes without raising.
    # Scenario (needs a TPU): two features (a and b) share one table, each given a dense float
    # id vector of batch size times chip count. Calls the static stats computation with these
    # dense features, the feature configuration, chip count and an SGD optimizer, and discards
    # the result.
    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    batch_size = 16
    num_tpu_chips = strategy.num_replicas_in_sync

    shared_table = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=self.embedding_dim,
        initializer=init_ops_v2.Constant(1.0),
        combiner='sum',
        name='shared_table',
    )

    features = ['feature_a', 'feature_b']
    feature_config = {
        feature: tpu_embedding_v2_utils.FeatureConfig(
            table=shared_table, name=feature, output_shape=[batch_size]
        )
        for feature in features
    }

    data = {
        feature: math_ops.cast(
            math_ops.range(batch_size * num_tpu_chips) % self.vocabulary_size,
            dtypes.float32,
        )
        for feature in features
    }

    tpu_embedding_v3.TPUEmbeddingV2.compute_sparse_core_stats(
        features=data,
        feature_config=feature_config,
        num_tpu_chips=num_tpu_chips,
        optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
    )

  def test_raise_error_when_weight_decay_is_set(self):
    # [FUNCTION DOC]
    # Scenario (needs a TPU): constructs the embedding layer inside a strategy scope with an SGD
    # optimizer that has a weight decay factor of 0.1 multiplied by the learning rate. Asserts
    # that construction raises a not-implemented error, because weight decay is unsupported.
    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_video, name='watched', output_shape=[16]
    )

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    with self.assertRaises(NotImplementedError):
      with strategy.scope():
        tpu_embedding_v3.TPUEmbeddingV2(
            feature_config=feature_config,
            optimizer=tpu_embedding_v2_utils.SGD(
                learning_rate=1.0,
                weight_decay_factor=0.1,
                multiply_weight_decay_factor_by_learning_rate=True,
            ),
        )

  @parameterized.parameters(
      *list(
          itertools.product(
              [True, False], [True, False], [True, False], [True, False]
          )
      )
  )
  def test_pipelining_and_summary_recording(
      self,
      record_summaries,
      is_enqueue_separated,
      use_predicate_function,
      use_sequential_embedding_context,
  ):
    # [FUNCTION DOC]
    # Parameterized over all combinations of four booleans: record summaries, enqueue separated
    # from the step, summary predicate given as a function, and use of a sequential embedding
    # context. Scenario (needs a TPU): builds a layer with pipelining enabled and a TPU function
    # running ten steps (either a combined lookup step or a separate enqueue plus dequeue),
    # applying gradients each step. Under a mock summary writer and a record-if condition it
    # asserts that the function is flagged for pipelining only when summaries are not recorded
    # (a predicate function always allows pipelining, a sequential embedding context never does).
    # The mock writer's init, flush and close do nothing.
    """Test that pipelining is disabled when summaries are recorded and vice versa."""
    num_steps = 10
    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_video, name='watched', output_shape=[16]
    )

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(1024)],
            values=np.arange(1024),
            dense_shape=[16, 1024],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    tpu_iter = iter(dist)

    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
          pipeline_execution_with_tensor_core=True,
      )
      weight = tf_variables.Variable(initial_value=10)

    if is_enqueue_separated:

      @def_function.function
      def tpu_test_fn(tpu_iter):
        # [FUNCTION DOC]
        # Graph function taking a distributed iterator. Runs ten steps in a loop; each step
        # enqueues the next batch on the host (separate function) and runs a per-replica step
        # that dequeues. Returns the activations of the last step.
        def step(data):
          # [FUNCTION DOC]
          # Per-replica step: dequeues activations and preserved results for the enqueued
          # data, scales the activations by the weight variable, applies them as gradients, and
          # returns the scaled activations.
          activations, preserved_results = mid_level_api.dequeue(data)
          activations = activations * weight
          mid_level_api.apply_gradients(activations, preserved_results)
          return activations

        @def_function.function
        def enqueue(data):
          # [FUNCTION DOC]
          # Graph function that enqueues one batch on the first worker device and returns the
          # partitioned host tensors.
          return mid_level_api.enqueue(
              data, device=strategy.extended.worker_devices[0]
          )

        inner_result = array_ops.zeros(
            [16, self.embedding_dim], dtype=dtypes.float32
        )
        for _ in math_ops.range(num_steps):
          host_partition_result = enqueue(next(tpu_iter))
          inner_result = strategy.run(step, args=(host_partition_result,))
        return inner_result

    else:

      @def_function.function
      def tpu_test_fn(tpu_iter):
        # [FUNCTION DOC]
        # Graph function taking a distributed iterator. Runs ten steps in a loop; each step
        # runs the combined lookup and gradient step on the next batch. Returns the activations
        # of the last step.
        def step(data):
          # [FUNCTION DOC]
          # Per-replica step: performs the lookup (returning activations and preserved
          # results), scales the activations by the weight variable, applies them as gradients,
          # and returns the scaled activations.
          activations, preserved_results = mid_level_api(data)
          activations = activations * weight
          mid_level_api.apply_gradients(activations, preserved_results)
          return activations

        inner_result = array_ops.zeros(
            [16, self.embedding_dim], dtype=dtypes.float32
        )
        for _ in math_ops.range(num_steps):
          inner_result = strategy.run(step, args=(next(tpu_iter),))
        return inner_result

    class MockSummaryWriter(summary_ops_v2.SummaryWriter):
      # [CLASS DOC]
      # HELPER: stub summary writer used only to make summary recording active in the test. It
      # overrides init, flush and close to do nothing, so no files are created.

      def init(self):
        # [FUNCTION DOC]
        # Stub: does nothing and returns nothing.
        pass

      def flush(self):
        # [FUNCTION DOC]
        # Stub: does nothing and returns nothing.
        pass

      def close(self):
        # [FUNCTION DOC]
        # Stub: does nothing and returns nothing.
        pass

    if use_predicate_function:
      should_record_summaries = lambda: record_summaries
      should_be_pipelined = True
    else:
      should_record_summaries = record_summaries
      should_be_pipelined = not record_summaries

    sequential_embedding_context = contextlib.nullcontext()
    if use_sequential_embedding_context:
      should_be_pipelined = False
      sequential_embedding_context = ecu.SequentialEmbeddingContext()

    with MockSummaryWriter().as_default():
      with sequential_embedding_context:
        with summary_ops_v2.record_if(should_record_summaries):
          is_pipelined = self.is_function_pipelined(tpu_test_fn, tpu_iter)
          # If we are recording summaries the function should not be pipelined
          # and vice versa.
          self.assertEqual(is_pipelined, should_be_pipelined)

  def is_function_pipelined(self, tf_function, *args):
    # [FUNCTION DOC]
    # HELPER: inspects a traced function to decide whether embedding pipelining is enabled.
    # KNOWN ISSUE: the docstring's Returns section is followed by further explanation text, and
    # the word "usage" in its example heading is misspelled; behavior matches otherwise.
    # Inputs: tf_function, a graph function; args, the arguments used to trace it.
    # Behavior: traces a concrete function, finds the operation named while in its graph (fails
    # the test if absent), obtains the loop body function (through the concrete function lookup
    # or, on attribute error, by scanning the graph's registered functions by name; raises a
    # value error if not found), then looks through the body operations for the pipelining
    # attribute, logging the first found.
    # Returns true if the attribute value is the forward or backward pipeline mode, else false.
    # Side effects: traces (and may compile) the function and writes an informational log line.
    """Returns whether the tf_function is flagged for embedding pipelining.

    Args:
      tf_function: a tf.function.
      *args: the arguments to the tf_function.

    Returns:
      Whether the tf_function is (will be) pipelined.

    This helper looks for a while loop in the provided function. It then looks
    for any op that has the pipelining attribute (e.g.,
    XlaSparseDenseMatmulWithCsrInput). The presence of the attribute indicates
    that the function is to be pipelined during compilation.

    Example usge:

      with summary_ops_v2.record_if(False):
        is_pipelined = self.is_function_pipelined(tpu_test_fn, tpu_iter)
        self.assertTrue(is_pipelined)
      with summary_ops_v2.record_if(True):
        is_pipelined = self.is_function_pipelined(tpu_test_fn, tpu_iter)
        self.assertFalse(is_pipelined)
    """
    # The attribute to look for to see if function is pipelined:
    attr_name = tpu_embedding_v3._PIPELINE_ATTRIBUTE

    ############################################################################
    # First, find the while loop op.
    ############################################################################
    func_graph = tf_function.get_concrete_function(*args).graph
    # type(func_graph) is tensorflow.python.framework.func_graph.FuncGraph
    while_op = None
    for op in func_graph.get_operations():
      # type(op) is tensorflow.python.framework.ops.Operation
      if op.name == 'while':
        while_op = op
        break
    self.assertIsNotNone(while_op, 'while op not found')

    ############################################################################
    # Second, find the body of the while loop.
    ############################################################################
    body_name = while_op.get_attr('body').name
    while_body_func = None
    try:
      while_body_func = func_graph.get_concrete_function(body_name)
    except AttributeError as exc:
      for func in while_op.graph._functions.values():
        if func.name.decode() == body_name:
          while_body_func = func
          break
      if while_body_func is None:
        raise ValueError('body not found') from exc
    while_body_graph = while_body_func.graph

    ############################################################################
    # Finally, find any pipelined SC op by looking for a pipelining attribute.
    ############################################################################
    attr_value = None
    for op in while_body_graph.get_operations():
      try:
        attr = op.get_attr(attr_name)
        logging.info(
            'Op "%s" has pipelining attr: %s : %s', op.name, attr_name, attr
        )
        attr_value = attr.decode('utf-8')
        break
      except ValueError:
        pass
    has_pipelining_attr = attr_value in [
        tpu_embedding_v3._PIPELINE_MODE_FORWARD,
        tpu_embedding_v3._PIPELINE_MODE_BACKWARD,
    ]
    return has_pipelining_attr


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  config.enable_mlir_bridge()
  test.main()
