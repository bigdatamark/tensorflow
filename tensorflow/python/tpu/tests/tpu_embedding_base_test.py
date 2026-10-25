# [MODULE DOC]
# HELPER: shared base module for the TPU embedding test files in this directory.
# Provides the base test case used by the TPU embedding (mid-level API) tests. Defines absl command
# line flags at import time for the TPU name, GCP project, GCP zone and a model directory (default
# taken from the test temp directory environment variable). Contains only the base class, no tests.
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
"""Base Class for TPU Embedding tests."""

import os
from typing import Tuple

from absl import flags
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import remote
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.util import nest


FLAGS = flags.FLAGS
flags.DEFINE_string('tpu', '', 'Name of TPU to connect to.')
flags.DEFINE_string('project', None, 'Name of GCP project with TPU.')
flags.DEFINE_string('zone', None, 'Name of GCP zone with TPU.')
flags.DEFINE_string('model_dir', os.environ.get('TEST_TMPDIR'),
                    'A temporary directory.')


class TPUEmbeddingBaseTest(parameterized.TestCase, test.TestCase):
  # [CLASS DOC]
  # HELPER: common fixture and utility base class for the TPU embedding test suites.
  # Test case base (parameterized plus the framework test case) that builds a small embedding setup in
  # its set-up: a video table (8 rows, dim 4, sum combiner), a user table (16 rows, dim 2, mean combiner),
  # three features (watched, favorited, friends) and fixed sparse inputs and row lengths, plus
  # high-dimensional variants. Offers helpers to connect to a TPU and create a strategy, build the
  # mid-level embedding object for a named optimizer, build sparse/ragged/dense datasets, and verify
  # activations, loss, gradients, embedding variables and optimizer slot variables against numpy
  # reference computations. Instance state: embedding_values, initializer, tables, feature_config,
  # batch sizes, index/value/row-length lists per feature and the cluster resolver.

  def skip_if_oss(self):
    # [FUNCTION DOC]
    # Skips the current test when either the project or the zone flag has been supplied.
    # Inputs: none besides the test instance. Reads the flags; if a project or zone is set (meaning a
    # cloud TPU is in use) it marks the test skipped with a message about slowness. Returns nothing.
    # KNOWN ISSUE: the name suggests skipping in open source builds, but the condition actually triggers
    # whenever a cloud project or zone flag is given.
    if FLAGS.project is not None or FLAGS.zone is not None:
      self.skipTest(
          'Skipping tests for oss as it is slow to run every test in cloud tpu.'
      )

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture set-up. Calls the parent set-up, then creates the embedding values (numbers 0 to 31 as
    # float64), a constant initializer, the video and user table configurations, and the feature
    # configuration tuple (watched and favorited use the video table, friends uses the user table).
    # Sets batch size 2 and data batch size 4, defines the indices, values and row lengths for the three
    # features (documented in comments as a global batch), sets the resolver to nothing, and builds
    # high-dimensional versions of indices (a leading batch index column added and repeated per data
    # batch), values and row lengths (repeated data batch size times). Returns nothing; only sets
    # instance attributes.
    super(TPUEmbeddingBaseTest, self).setUp()
    self.embedding_values = np.array(list(range(32)), dtype=np.float64)
    self.initializer = init_ops_v2.Constant(self.embedding_values)
    # Embedding for video initialized to
    # 0 1 2 3
    # 4 5 6 7
    # ...
    self.table_video = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=8,
        dim=4,
        initializer=self.initializer,
        combiner='sum',
        name='video')
    # Embedding for user initialized to
    # 0 1
    # 2 3
    # 4 5
    # 6 7
    # ...
    self.table_user = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=16,
        dim=2,
        initializer=self.initializer,
        combiner='mean',
        name='user')
    self.feature_config = (tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_video, name='watched'),
                           tpu_embedding_v2_utils.FeatureConfig(
                               table=self.table_video, name='favorited'),
                           tpu_embedding_v2_utils.FeatureConfig(
                               table=self.table_user, name='friends'))

    self.batch_size = 2
    self.data_batch_size = 4

    # One (global) batch of inputs
    # sparse tensor for watched:
    # row 0: 0
    # row 1: 0, 1
    # row 2: 0, 1
    # row 3: 1
    self.feature_watched_indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1],
                                    [3, 0]]
    self.feature_watched_values = [0, 0, 1, 0, 1, 1]
    self.feature_watched_row_lengths = [1, 2, 2, 1]
    # sparse tensor for favorited:
    # row 0: 0, 1
    # row 1: 1
    # row 2: 0
    # row 3: 0, 1
    self.feature_favorited_indices = [[0, 0], [0, 1], [1, 0], [2, 0], [3, 0],
                                      [3, 1]]
    self.feature_favorited_values = [0, 1, 1, 0, 0, 1]
    self.feature_favorited_row_lengths = [2, 1, 1, 2]
    # sparse tensor for friends:
    # row 0: 3
    # row 1: 0, 1, 2
    # row 2: 3
    # row 3: 0, 1, 2
    self.feature_friends_indices = [[0, 0], [1, 0], [1, 1], [1, 2], [2, 0],
                                    [3, 0], [3, 1], [3, 2]]
    self.feature_friends_values = [3, 0, 1, 2, 3, 0, 1, 2]
    self.feature_friends_row_lengths = [1, 3, 1, 3]
    self.resolver = None

    # Basically we are expand the dims of the old feature by 1 and repeat
    # batch size times for the first dimension.
    def create_hight_dimensional_indices(indices):
      # [FUNCTION DOC]
      # HELPER: nested utility of setUp that builds the indices for the high-dimensional features.
      # Input: indices, a list of two-element index pairs. Converts them to an int32 array, prepends a
      # column holding the batch position (each repeated once per original index), tiles the original
      # indices once per data batch element, and returns the concatenated array of three-column indices.
      indices = np.array(indices, dtype=np.int32)
      batch_size_index = np.repeat(
          np.arange(self.data_batch_size), len(indices)).reshape(-1, 1)
      repeated_indices = np.tile(indices, (self.data_batch_size, 1))
      return np.concatenate([batch_size_index, repeated_indices], axis=1)

    # Create high dimensional features with shape(4, 4, 2)
    self.feature_watched_indices_high_dimensional = create_hight_dimensional_indices(
        self.feature_watched_indices)
    self.feature_watched_values_high_dimensional = self.feature_watched_values * self.data_batch_size
    self.feature_watched_row_lengths_high_dimensional = self.feature_watched_row_lengths * self.data_batch_size

    # Create high dimensional features with shape(4, 4, 2)
    self.feature_favorited_indices_high_dimensional = create_hight_dimensional_indices(
        self.feature_favorited_indices)
    self.feature_favorited_values_high_dimensional = self.feature_favorited_values * self.data_batch_size
    self.feature_favorited_row_lengths_high_dimensional = self.feature_favorited_row_lengths * self.data_batch_size

    # Create high dimensional features with shape(4, 4, 3)
    self.feature_friends_indices_high_dimensional = create_hight_dimensional_indices(
        self.feature_friends_indices)
    self.feature_friends_values_high_dimensional = self.feature_friends_values * self.data_batch_size
    self.feature_friends_row_lengths_high_dimensional = self.feature_friends_row_lengths * self.data_batch_size

  def _init_tpu_system(self):
    # [FUNCTION DOC]
    # HELPER: connects the test to the TPU system.
    # Creates a TPU cluster resolver from the tpu, zone and project flags and stores it on the instance.
    # If the resolver exposes a cloud TPU client, configures the TPU software version to nightly with
    # restart always. Connects to the cluster and initializes the TPU system. Returns the result of
    # the initialization. Side effects: remote connection, TPU restart and initialization.
    self.resolver = tpu_cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu, zone=FLAGS.zone, project=FLAGS.project)
    if hasattr(self.resolver, '_cloud_tpu_client'):
      self.resolver._cloud_tpu_client.configure_tpu_version(
          version='nightly', restart_type='always')
    remote.connect_to_cluster(self.resolver)
    return tpu_cluster_resolver.initialize_tpu_system(self.resolver)

  def _get_strategy(self):
    # [FUNCTION DOC]
    # HELPER: creates the distribution strategy for tests.
    # Initializes the TPU system (result ignored) and returns a TPU strategy built on the stored resolver.
    _ = self._init_tpu_system()
    return tpu_strategy.TPUStrategy(self.resolver)

  def _create_mid_level(self, optimizer=None):
    # [FUNCTION DOC]
    # HELPER: builds the mid-level TPU embedding object for the test's feature configuration.
    # Input: optimizer, optional; when nothing is given uses plain gradient descent with learning rate
    # 0.1. Returns a TPU embedding object created with the feature config and the optimizer.
    # Create `TPUEmbedding` object.
    if optimizer is None:
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)

    return tpu_embedding_v2.TPUEmbedding(
        feature_config=self.feature_config, optimizer=optimizer)

  def _create_strategy_and_mid_level(self, optimizer_name) -> Tuple[
      tpu_strategy.TPUStrategy, tpu_embedding_v2.TPUEmbedding,
      tpu_embedding_v2_utils._Optimizer]:
    # [FUNCTION DOC]
    # HELPER: creates strategy, embedding object and optimizer in one step.
    # Input: optimizer_name, one of sgd, adagrad, adam, ftrl or adagrad_momentum (all with learning rate
    # 0.1; the momentum one uses momentum 0.9, nesterov, exponent 3, epsilon 0.1, beta2 0.9).
    # Gets a strategy, and inside its scope builds the optimizer and the mid-level embedding object.
    # Returns a tuple of strategy, embedding object and optimizer. Raises a value error for an unknown
    # optimizer name.
    strategy = self._get_strategy()

    with strategy.scope():
      if optimizer_name == 'sgd':
        optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      elif optimizer_name == 'adagrad':
        optimizer = tpu_embedding_v2_utils.Adagrad(learning_rate=0.1)
      elif optimizer_name == 'adam':
        optimizer = tpu_embedding_v2_utils.Adam(learning_rate=0.1)
      elif optimizer_name == 'ftrl':
        optimizer = tpu_embedding_v2_utils.FTRL(learning_rate=0.1)
      elif optimizer_name == 'adagrad_momentum':
        optimizer = tpu_embedding_v2_utils.AdagradMomentum(
            learning_rate=0.1,
            momentum=0.9,
            use_nesterov=True,
            exponent=3.0,
            epsilon=0.1,
            beta2=0.9)
      else:
        raise ValueError('optimizer is not recognized: ', optimizer_name)
      mid_level_api = self._create_mid_level(optimizer=optimizer)

    return strategy, mid_level_api, optimizer

  def _create_sparse_data(self, include_weights, weight=0.5):
    # [FUNCTION DOC]
    # HELPER: builds the fixed sparse input features for tests.
    # Inputs: include_weights, whether to add weights; weight, constant weight value (default 0.5).
    # Returns a tuple of three sparse tensors (watched, favorited, friends) with dense shape of data batch
    # size by 2, 2 and 3. If weights are requested, returns a pair of the features and a tuple of sparse
    # tensors with the same indices whose values are all the given weight as float32.
    sparse_features = (sparse_tensor.SparseTensor(
        indices=self.feature_watched_indices,
        values=self.feature_watched_values,
        dense_shape=[self.data_batch_size, 2]),
                       sparse_tensor.SparseTensor(
                           indices=self.feature_favorited_indices,
                           values=self.feature_favorited_values,
                           dense_shape=[self.data_batch_size, 2]),
                       sparse_tensor.SparseTensor(
                           indices=self.feature_friends_indices,
                           values=self.feature_friends_values,
                           dense_shape=[self.data_batch_size, 3]))
    if include_weights:
      weights = []
      for sparse in sparse_features:
        values = (
            array_ops.ones_like(sparse.values, dtype=dtypes.float32) * weight)
        weights.append(
            sparse_tensor.SparseTensor(
                indices=sparse.indices,
                values=values,
                dense_shape=sparse.dense_shape))
      sparse_features = (sparse_features, tuple(weights))
    return sparse_features

  def _create_sparse_dataset(self, strategy, include_weights=False, weight=0.5):
    # [FUNCTION DOC]
    # HELPER: builds a dataset of sparse inputs for enqueueing.
    # Inputs: strategy, used for its replica count; include_weights and weight as for the sparse data.
    # Makes a dataset from the sparse features, unbatches, repeats forever, and rebatches to batch size
    # times number of replicas, dropping remainder. Returns the dataset.
    # Create dataset for enqueue operation
    sparse_features = self._create_sparse_data(include_weights, weight)

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)

    # Data is batched to self.data_batch_size, rebatch to global batch size.
    return dataset.unbatch().repeat().batch(
        self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

  def _create_high_dimensional_sparse_dataset(self,
                                              strategy,
                                              include_weights=False,
                                              weight=0.5):
    # [FUNCTION DOC]
    # HELPER: builds a dataset of sparse inputs with an extra dimension.
    # Inputs: strategy (replica count), include_weights and weight (default 0.5). Builds three sparse
    # tensors from the high-dimensional indices and values, with dense shapes of data batch size by data
    # batch size by 2, 2 and 3, optionally adds same-shaped constant weights, and returns a dataset that
    # is unbatched, repeated forever and rebatched to the global batch size, dropping the remainder.
    sparse_features = (
        sparse_tensor.SparseTensor(
            indices=self.feature_watched_indices_high_dimensional,
            values=self.feature_watched_values_high_dimensional,
            dense_shape=[self.data_batch_size, self.data_batch_size, 2]),
        sparse_tensor.SparseTensor(
            indices=self.feature_favorited_indices_high_dimensional,
            values=self.feature_favorited_values_high_dimensional,
            dense_shape=[self.data_batch_size, self.data_batch_size, 2]),
        sparse_tensor.SparseTensor(
            indices=self.feature_friends_indices_high_dimensional,
            values=self.feature_friends_values_high_dimensional,
            dense_shape=[self.data_batch_size, self.data_batch_size, 3]))
    if include_weights:
      weights = []
      for sparse in sparse_features:
        values = (
            array_ops.ones_like(sparse.values, dtype=dtypes.float32) * weight)
        weights.append(
            sparse_tensor.SparseTensor(
                indices=sparse.indices,
                values=values,
                dense_shape=sparse.dense_shape))
      sparse_features = (sparse_features, tuple(weights))

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    # Data is batched to self.data_batch_size, rebatch to global batch size.
    return dataset.unbatch().repeat().batch(
        self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

  def _create_high_dimensional_ragged_dataset(self,
                                              strategy,
                                              include_weights=False,
                                              weight=0.5):
    # [FUNCTION DOC]
    # HELPER: builds a dataset of ragged inputs with an extra dimension.
    # Inputs: strategy (replica count), include_weights and weight (default 0.5). Builds ragged tensors
    # from the high-dimensional row lengths and values for the three features, optionally adds ragged
    # weights with the same row lengths, and returns the dataset unbatched, repeated and rebatched to the
    # global batch size, dropping the remainder.
    # KNOWN ISSUE: weights are created through the plain ragged tensor constructor with row lengths and
    # values, which is not the usual factory-method route and may not be a valid construction.
    ragged_features = (
        ragged_tensor.RaggedTensor.from_row_lengths(
            row_lengths=self.feature_watched_row_lengths_high_dimensional,
            values=self.feature_watched_values_high_dimensional),
        ragged_tensor.RaggedTensor.from_row_lengths(
            row_lengths=self.feature_favorited_row_lengths_high_dimensional,
            values=self.feature_favorited_values_high_dimensional),
        ragged_tensor.RaggedTensor.from_row_lengths(
            row_lengths=self.feature_friends_row_lengths_high_dimensional,
            values=self.feature_friends_values_high_dimensional))
    if include_weights:
      weights = []
      for ragged in ragged_features:
        values = (
            array_ops.ones_like(ragged.values, dtype=dtypes.float32) * weight)
        weights.append(
            ragged_tensor.RaggedTensor(
                row_lengths=ragged.row_lengths(), values=values))
      ragged_features = (ragged_features, tuple(weights))

    dataset = dataset_ops.DatasetV2.from_tensors(ragged_features)
    # Data is batched to self.data_batch_size, rebatch to global batch size.
    return dataset.unbatch().repeat().batch(
        self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

  def _create_ragged_dataset(self, strategy, include_weights=False, weight=0.5):
    # [FUNCTION DOC]
    # HELPER: builds a dataset of ragged inputs.
    # Inputs: strategy (replica count), include_weights and weight (default 0.5). Takes the sparse
    # features (and weights) and converts each to a ragged tensor, then returns a dataset that is
    # unbatched, repeated and rebatched to batch size times replicas, dropping the remainder.
    # Create dataset for enqueue operation
    sparse_features = self._create_sparse_data(include_weights, weight)
    ragged_features = nest.map_structure(ragged_tensor.RaggedTensor.from_sparse,
                                         sparse_features)

    dataset = dataset_ops.DatasetV2.from_tensors(ragged_features)

    # Data is batched to self.data_batch_size, rebatch to global batch size.
    return dataset.unbatch().repeat().batch(
        self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

  def _create_dense_dataset(self, strategy, include_weights=False, weight=0.5):
    # [FUNCTION DOC]
    # HELPER: builds a dataset of dense id inputs.
    # Inputs: strategy (replica count), include_weights and weight (default 0.5). Features are int32
    # constants taken from the first data-batch-size values of each feature; optional weights are float
    # tensors filled with the weight. Returns the unbatched, repeated and rebatched dataset.

    features = (constant_op.constant(
        self.feature_watched_values[:self.data_batch_size], dtype=dtypes.int32),
                constant_op.constant(
                    self.feature_favorited_values[:self.data_batch_size],
                    dtype=dtypes.int32),
                constant_op.constant(
                    self.feature_friends_values[:self.data_batch_size],
                    dtype=dtypes.int32))
    if include_weights:
      weights = [
          array_ops.ones_like(t, dtype=dtypes.float32) * weight
          for t in features
      ]
      features = (features, tuple(weights))

    dataset = dataset_ops.DatasetV2.from_tensors(features)
    return dataset.unbatch().repeat().batch(
        self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

  def _create_high_dimensional_dense_dataset(self,
                                             strategy,
                                             include_weights=False,
                                             weight=0.5):
    # [FUNCTION DOC]
    # HELPER: builds a dataset of dense id inputs with an extra dimension.
    # Inputs: strategy (replica count), include_weights and weight (default 0.5). Features are int32
    # constants of shape data batch size by data batch size by 1 built from the high-dimensional values;
    # optional constant weights are added. Returns the unbatched, repeated, rebatched dataset.

    dense_size = self.data_batch_size * self.data_batch_size
    features = (constant_op.constant(
        self.feature_watched_values_high_dimensional[:dense_size],
        shape=(self.data_batch_size, self.data_batch_size, 1),
        dtype=dtypes.int32),
                constant_op.constant(
                    self.feature_favorited_values_high_dimensional[:dense_size],
                    shape=(self.data_batch_size, self.data_batch_size, 1),
                    dtype=dtypes.int32),
                constant_op.constant(
                    self.feature_friends_values_high_dimensional[:dense_size],
                    shape=(self.data_batch_size, self.data_batch_size, 1),
                    dtype=dtypes.int32))
    if include_weights:
      weights = [
          array_ops.ones_like(t, dtype=dtypes.float32) * weight
          for t in features
      ]
      features = (features, tuple(weights))
    dataset = dataset_ops.DatasetV2.from_tensors(features)
    return dataset.unbatch().repeat().batch(
        self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

  def _check_results(self, strategy, shard_out_val, training, input_data,
                     table_to_variable, optimizer, is_high_dimensional):
    # [FUNCTION DOC]
    # HELPER: verification routine shared by the correctness tests.
    # Inputs: strategy; shard_out_val, per-replica outputs (loss then the three activations); training,
    # whether to also check updated variables; input_data, the sparse inputs used; table_to_variable,
    # mapping from table name to its variables; optimizer; is_high_dimensional flag. Unpacks the outputs,
    # builds gold activations from the known initial table values for core 0 and the other cores, expands
    # them per replica (stacked for high-dimensional input), computes the gold loss, and asserts the
    # activations and the loss are close. When training, computes expected gradients with respect to each
    # table and checks the embedding and slot variables. Returns nothing; failures are assertion errors.
    num_replicas = strategy.num_replicas_in_sync

    # Unpack the values `strategy.run()` returns.
    loss = self._unpack(strategy, shard_out_val[0])
    activation_watched = self._unpack(strategy, shard_out_val[1])
    activation_favorited = self._unpack(strategy, shard_out_val[2])
    activation_friends = self._unpack(strategy, shard_out_val[3])

    # Core 0:
    # Calculate the values of embedding activations.
    activation_watched_gold0 = np.array([[0, 1, 2, 3], [4, 6, 8, 10]])
    activation_favorited_gold0 = np.array([[4, 6, 8, 10], [4, 5, 6, 7]])
    # Second row of `activation_friends_gold0` is the mean of the following.
    # row 0: 0 1
    # row 1: 2 3
    # row 2: 4 5
    activation_friends_gold0 = np.array([[6, 7], [2, 3]])

    loss_gold0 = self._compute_loss(activation_watched_gold0,
                                    activation_favorited_gold0,
                                    activation_friends_gold0)

    # Add on values from other cores:
    # Activations for watched are an alternating sequence of
    # activation_watched_gold0 and activation_favorited_gold0.
    # For favorited it is the same but in the opposite order.
    activation_watched_gold = np.concatenate(
        (activation_watched_gold0, activation_favorited_gold0))
    activation_favorited_gold = np.concatenate(
        (activation_favorited_gold0, activation_watched_gold0))
    activation_friends_gold = np.concatenate(
        (activation_friends_gold0, activation_friends_gold0))

    if is_high_dimensional:
      activation_watched_gold = np.stack([activation_watched_gold] *
                                         self.batch_size * num_replicas)

      activation_favorited_gold = np.stack([activation_favorited_gold] *
                                           self.batch_size * num_replicas)

      activation_friends_gold = np.stack([activation_friends_gold] *
                                         self.batch_size * num_replicas)
    else:
      if num_replicas == 1:
        activation_watched_gold = activation_watched_gold0
        activation_favorited_gold = activation_favorited_gold0
        activation_friends_gold = activation_friends_gold0
      else:
        activation_watched_gold = np.concatenate(
            [activation_watched_gold] * (num_replicas // self.batch_size))
        activation_favorited_gold = np.concatenate(
            [activation_favorited_gold] * (num_replicas // self.batch_size))
        activation_friends_gold = np.concatenate(
            [activation_friends_gold] * (num_replicas // self.batch_size))

    loss_gold = [loss_gold0] * num_replicas

    # Test values.
    self.assertAllClose(activation_watched_gold, activation_watched)
    self.assertAllClose(activation_favorited_gold, activation_favorited)
    self.assertAllClose(activation_friends_gold, activation_friends)

    self.assertAllClose(loss_gold, loss)

    embedding_table_video_before = np.copy(
        np.reshape(self.embedding_values, [8, 4]))
    embedding_table_user_before = np.copy(
        np.reshape(self.embedding_values, [16, 2]))
    if is_high_dimensional:
      global_batch_size = self.batch_size * self.data_batch_size * num_replicas
    else:
      global_batch_size = self.batch_size * num_replicas
    if training:
      gradient_wrt_watched_gold = (2 * activation_watched_gold /
                                   global_batch_size)
      gradient_wrt_favorited_gold = (2 * activation_favorited_gold /
                                     global_batch_size)
      gradient_wrt_friends_gold = (2 * activation_friends_gold /
                                   global_batch_size)

      # Calculate gradients wrt embedding tables.
      gradients_wrt_user = (
          self._compute_gradients_wrt_embedding_table(
              gradient_wrt_friends_gold, embedding_table_user_before,
              input_data[2].indices.numpy(), input_data[2].values.numpy(),
              self.table_user.combiner))
      gradients_wrt_video = (
          self._compute_gradients_wrt_embedding_table(
              gradient_wrt_favorited_gold, embedding_table_video_before,
              input_data[1].indices.numpy(), input_data[1].values.numpy(),
              self.table_video.combiner) +
          self._compute_gradients_wrt_embedding_table(
              gradient_wrt_watched_gold, embedding_table_video_before,
              input_data[0].indices.numpy(), input_data[0].values.numpy(),
              self.table_video.combiner))

      self._check_embedding_and_slot_variables(embedding_table_user_before,
                                               gradients_wrt_user,
                                               embedding_table_video_before,
                                               gradients_wrt_video, optimizer,
                                               table_to_variable)

  def _check_embedding_and_slot_variables(self, embedding_table_user_before,
                                          gradients_wrt_user,
                                          embedding_table_video_before,
                                          gradients_wrt_video, optimizer,
                                          table_to_variable):
    # [FUNCTION DOC]
    # HELPER: dispatches variable checks by optimizer type.
    # Inputs: tables' values before the update and gradients for user and video tables, the optimizer, and
    # the mapping from table name to variables. Picks the check routine matching the optimizer class
    # (plain gradient descent, adagrad, adagrad momentum, adam, ftrl) and applies it to both tables.
    # Raises a value error when the optimizer type is not recognized.
    if isinstance(optimizer, tpu_embedding_v2_utils.SGD):
      check_fn = self._check_embedding_and_slot_variables_for_sgd
    elif isinstance(optimizer, tpu_embedding_v2_utils.Adagrad):
      check_fn = self._check_embedding_and_slot_variables_for_adagrad
    elif isinstance(optimizer, tpu_embedding_v2_utils.AdagradMomentum):
      check_fn = self._check_embedding_and_slot_variables_for_adagrad_momentum
    elif isinstance(optimizer, tpu_embedding_v2_utils.Adam):
      check_fn = self._check_embedding_and_slot_variables_for_adam
    elif isinstance(optimizer, tpu_embedding_v2_utils.FTRL):
      check_fn = self._check_embedding_and_slot_variables_for_ftrl
    else:
      raise ValueError('optimizer is not recognized: ', type(optimizer))
    check_fn(embedding_table_user_before, gradients_wrt_user, optimizer,
             table_to_variable[self.table_user.name])
    check_fn(embedding_table_video_before, gradients_wrt_video, optimizer,
             table_to_variable[self.table_video.name])

  def _check_embedding_and_slot_variables_for_sgd(self, embedding_table_before,
                                                  gradients, optimizer,
                                                  variables):
    # [FUNCTION DOC]
    # HELPER: checks parameters after a gradient descent update.
    # Inputs: table values before, gradients, optimizer, variables mapping with a parameters entry.
    # Subtracts learning rate times the summed gradients from the old table and asserts the actual
    # parameter variable is close. Returns nothing.
    embedding_table = np.copy(embedding_table_before)
    embedding_table -= optimizer.learning_rate * np.sum(gradients, axis=0)
    self.assertAllClose(
        self._get_variable(variables['parameters']).numpy(), embedding_table)

  def _check_embedding_and_slot_variables_for_adagrad(self,
                                                      embedding_table_before,
                                                      gradients, optimizer,
                                                      variable):
    # [FUNCTION DOC]
    # HELPER: checks parameters and accumulators after an adagrad update.
    # Inputs: table before, gradients, optimizer, variable mapping (parameters and accumulators).
    # Expected accumulator is the initial value plus the squared summed gradients; expected parameters
    # subtract learning rate times summed gradient divided by square root of the accumulator. Asserts both
    # variables are close. Returns nothing.
    embedding_table = np.copy(embedding_table_before)
    accumulator = (
        optimizer.initial_accumulator_value + np.sum(gradients, axis=0)**2)
    embedding_table -= (
        optimizer.learning_rate * np.sum(gradients, axis=0) /
        np.sqrt(accumulator))
    self.assertAllClose(
        self._get_variable(variable['parameters']).numpy(), embedding_table)
    self.assertAllClose(
        self._get_variable(variable['accumulators']).numpy(), accumulator)

  def _check_embedding_and_slot_variables_for_adagrad_momentum(
      self, embedding_table_before, gradients, optimizer, variable):
    # [FUNCTION DOC]
    # HELPER: checks parameters, accumulators and momenta after an adagrad-momentum update.
    # Inputs: table before, gradients, optimizer, variable mapping (parameters, accumulators, momenta).
    # Starts from zero accumulators and momenta sized like the variables, sums the gradients, updates the
    # accumulator (plain sum of squares if beta2 is one, otherwise exponential average), raises it to the
    # negative inverse exponent after adding epsilon, updates the momentum, uses a nesterov or plain
    # update, and asserts all three variables are close with relative tolerance 1e-3. Returns nothing.
    embedding_table = np.copy(embedding_table_before)
    accumulator = np.zeros(self._get_variable(variable['accumulators']).shape)
    momenta = np.zeros(self._get_variable(variable['momenta']).shape)
    gradients = np.sum(gradients, axis=0)
    if optimizer.beta2 == 1.0:
      accumulator += gradients**2
    else:
      accumulator = optimizer.beta2 * accumulator + (
          1 - optimizer.beta2) * gradients**2
    accumulator_power = np.power(accumulator + optimizer.epsilon,
                                 -1.0 / optimizer.exponent)
    momenta = optimizer.momentum * momenta + gradients * accumulator_power
    if optimizer.use_nesterov:
      update = optimizer.momentum * momenta + gradients * accumulator_power
    else:
      update = momenta
    embedding_table -= optimizer.learning_rate * update
    self.assertAllClose(
        self._get_variable(variable['parameters']).numpy(),
        embedding_table,
        rtol=1e-3)
    self.assertAllClose(
        self._get_variable(variable['accumulators']).numpy(),
        accumulator,
        rtol=1e-3)
    self.assertAllClose(
        self._get_variable(variable['momenta']).numpy(), momenta, rtol=1e-3)

  def _check_embedding_and_slot_variables_for_adam(self, embedding_table_before,
                                                   gradients, optimizer,
                                                   variable):
    # [FUNCTION DOC]
    # HELPER: checks parameters, momenta and velocities after an adam update.
    # Inputs: table before, gradients, optimizer, variable mapping (parameters, momenta, velocities).
    # Computes first and second moments from the summed gradient scaled by one minus beta, applies the
    # learning rate without any decay factor (TPU embeddings have none), and asserts closeness with
    # relative tolerance 1e-4. Returns nothing.
    embedding_table = np.copy(embedding_table_before)
    g = np.sum(gradients, axis=0)
    v = g**2 * (1 - optimizer.beta_2)
    m = g * (1 - optimizer.beta_1)
    epsilon = optimizer.epsilon
    # TPU Embeddings don't have the LR decay factor for Adam.
    lr_modifier = 1
    embedding_table -= (
        m * optimizer.learning_rate * lr_modifier / (np.sqrt(v) + epsilon))
    self.assertAllClose(
        self._get_variable(variable['parameters']).numpy(),
        embedding_table,
        rtol=1e-4)
    self.assertAllClose(
        self._get_variable(variable['momenta']).numpy(), m, rtol=1e-4)
    self.assertAllClose(
        self._get_variable(variable['velocities']).numpy(), v, rtol=1e-4)

  def _check_embedding_and_slot_variables_for_ftrl(self, embedding_table_before,
                                                   gradients, optimizer,
                                                   variable):
    # [FUNCTION DOC]
    # HELPER: checks parameters, linears and accumulators after an FTRL update.
    # Inputs: table before, gradients, optimizer, variable mapping (parameters, linears, accumulators).
    # Computes accumulator, sigma, linear and quadratic terms, derives expected parameters, and zeroes
    # actual parameters wherever the linear term is zero because untouched rows keep initial values.
    # Asserts closeness with relative tolerances 5e-5 (parameters) and 5e-4 (linears); accumulators with
    # default tolerance. Returns nothing.
    embedding_table = np.copy(embedding_table_before)
    neg_lr_p = -optimizer.learning_rate_power
    accumulator = (
        optimizer.initial_accumulator_value + np.sum(gradients, axis=0)**2)
    sigma = (accumulator**neg_lr_p - optimizer.initial_accumulator_value**
             neg_lr_p) / optimizer.learning_rate
    linear = np.sum(gradients, axis=0) - sigma * embedding_table
    quadratic = accumulator**neg_lr_p / optimizer.learning_rate
    embedding_table = -linear / quadratic
    actual_parameters = self._get_variable(variable['parameters']).numpy()
    # For entries where `linear` == 0, it is not worth comparing since the
    # initial values have not been touched yet and they will not agree with what
    # the actual values should be.
    actual_parameters *= (linear != 0.0)
    # FTRL has a bit more precision diff on parameters.
    self.assertAllClose(actual_parameters, embedding_table, rtol=5e-5)
    self.assertAllClose(
        self._get_variable(variable['linears']).numpy(), linear, rtol=5e-4)
    self.assertAllClose(
        self._get_variable(variable['accumulators']).numpy(), accumulator)

  def _get_replica_numpy(self, structured, strategy, replica_id):
    # [FUNCTION DOC]
    # HELPER: extracts one replica's values from a nested structure as numpy.
    # Inputs: structured, a nested structure of distributed values; strategy; replica_id, integer.
    # Maps a selection function over the structure and returns the same structure of numpy arrays.

    def select_replica(x):
      # [FUNCTION DOC]
      # HELPER: nested mapper of the replica extraction.
      # Input: x, a distributed value. Gets the strategy's local results; if there is one result it is
      # converted to numpy, otherwise the entry for the replica id is converted. Returns a numpy array.
      # KNOWN ISSUE: in the single-result case numpy conversion is called on the tuple of local results
      # itself rather than on its only element, which would fail for a tuple.
      x = strategy.experimental_local_results(x)
      if len(x) == 1:
        return x.numpy()
      return x[replica_id].numpy()

    return nest.map_structure(select_replica, structured)

  def _compute_gradients_wrt_embedding_table(self, gradient_wrt_activation,
                                             embedding_table, feature_indices,
                                             feature_values, combiner):
    # [FUNCTION DOC]
    # HELPER: numpy reference for embedding table gradients.
    # Inputs: gradient_wrt_activation, array of batch by dimension; embedding_table, array of vocabulary
    # by dimension; feature_indices and feature_values as used to build a sparse tensor; combiner, mean or
    # sum. Accumulates each activation gradient into the row of the looked-up vocabulary id per batch
    # entry, counts entries per batch position, divides by the count (minimum one) for the mean combiner,
    # and returns the gradients reshaped to a flat batch dimension followed by the table shape.
    # Raises a value error if the combiner is neither mean nor sum.
    """Compute gradients wrt embedding_table.

    Args:
      gradient_wrt_activation: `np.array` with shape `batch_size` by embedding
        `dimension`.
      embedding_table: `np.array` with shape `vocabulary_size` by embedding
        `dimension`.
      feature_indices: `indices` as used to construct `SparseTensor`.
      feature_values: `values` as used to construct `SparseTensor`.
      combiner: `String`, 'mean' or 'sum'.

    Returns:
      Gradients wrt `embedding_table`, an `np.array`s with shape
        `batch_size` by `vocabulary_size` by
        embedding `dimension`.

    Raises:
      ValueError: if `combiner` is not one of 'mean' or 'sum'.
    """
    if combiner not in ('mean', 'sum'):
      raise ValueError(
          '`combiner` must be mean or sum; got {}.'.format(combiner))
    grads_shape = gradient_wrt_activation.shape[:-1] + embedding_table.shape
    grads = np.zeros(shape=grads_shape)
    count = np.zeros(shape=grads_shape)
    for feature_indice, vocabulary_id in zip(feature_indices, feature_values):
      batch_index = tuple(feature_indice[:-1])
      grads[batch_index][vocabulary_id] += gradient_wrt_activation[batch_index]
      count[batch_index] += 1
    count[count == 0] = 1
    if combiner == 'mean':
      grads = grads / count
    return np.reshape(grads, (-1, *embedding_table.shape))

  def _unpack(self, strategy, per_replica_output):
    # [FUNCTION DOC]
    # HELPER: gathers per-replica results into one array.
    # Inputs: strategy, per_replica_output. Gets local results, concatenates along the first axis and
    # returns the numpy array.
    per_replica_output = strategy.experimental_local_results(per_replica_output)
    per_replica_output = array_ops.concat(per_replica_output, axis=0).numpy()
    return per_replica_output

  def _get_total_loss_tensor(self, activations):
    # [FUNCTION DOC]
    # HELPER: builds the total loss tensor from activations.
    # Input: activations, a list of tensors. For each, takes the mean over the batch of the sum over the
    # last axis of squared values, adds them, and returns the total as a one-element tensor.
    losses = []
    for activation in activations:
      losses.append(
          math_ops.reduce_mean(
              math_ops.reduce_sum(
                  gen_math_ops.squared_difference(activation, 0), axis=-1)))
    total_loss = array_ops.expand_dims_v2(sum(losses), 0)
    return total_loss

  def _compute_loss(self, activation_watched, activation_favorited,
                    activation_friends):
    # [FUNCTION DOC]
    # HELPER: numpy reference loss.
    # Inputs: the three activation arrays. For each computes mean over batch of the sum of squares over
    # the last axis and returns the sum of the three as a number.
    watched_loss = np.mean(np.sum(activation_watched**2, axis=-1))
    favorited_loss = np.mean(np.sum(activation_favorited**2, axis=-1))
    friends_loss = np.mean(np.sum(activation_friends**2, axis=-1))
    loss = watched_loss + favorited_loss + friends_loss
    return loss

  def _get_variable(self, variable):
    # [FUNCTION DOC]
    # HELPER: unwraps embedding variables.
    # Input: variable. If it is a TPU embedding variable returns its first underlying variable, otherwise
    # returns the input unchanged.
    if isinstance(variable, tpu_embedding_v2.TPUEmbeddingVariable):
      return variable.variables[0]
    return variable

  def _get_tmpdir(self, name, subdir=''):
    # [FUNCTION DOC]
    # HELPER: builds a directory path under the model directory flag.
    # Inputs: name, a directory name; subdir, optional extra component (default empty). Returns the path
    # joining the model directory flag, the name and the subdirectory if provided. Creates nothing.
    segments = [FLAGS.model_dir, name] + ([subdir] if subdir else [])
    return os.path.join(*segments)
