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
# [MODULE DOC]
# Unit tests for the CPU serving variant of the TPU embedding mid level API. Builds small video
# and user tables with constant initial values and checks dense, sparse, ragged, weighted,
# sequence and high dimensional lookups against numpy golden results, plus error cases.
# No import-time side effects; runs with eager v2 behavior when executed as a script.
"""Tests for TPU Embeddings mid level API on CPU."""

import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_for_serving
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.util import nest


class TPUEmbeddingForServingTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the serving embedding class on CPU. Shared fixtures (created in setUp):
  # two table configs, three feature configs, and one global batch of inputs described as
  # indices, values and row lengths for each feature.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Creates a constant-valued initializer, the video table (8 rows, dim 4, sum
    # combiner) and user table (16 rows, dim 2, mean combiner), the feature configs for watched,
    # favorited (video) and friends (user), and the sparse input data for a batch of four rows.
    # Stores them as attributes; returns nothing.
    super(TPUEmbeddingForServingTest, self).setUp()

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
    self.feature_config = (
        tpu_embedding_v2_utils.FeatureConfig(
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
    self.feature_watched_indices = [[0, 0], [1, 0], [1, 1],
                                    [2, 0], [2, 1], [3, 0]]
    self.feature_watched_values = [0, 0, 1, 0, 1, 1]
    self.feature_watched_row_lengths = [1, 2, 2, 1]
    # sparse tensor for favorited:
    # row 0: 0, 1
    # row 1: 1
    # row 2: 0
    # row 3: 0, 1
    self.feature_favorited_indices = [[0, 0], [0, 1], [1, 0],
                                      [2, 0], [3, 0], [3, 1]]
    self.feature_favorited_values = [0, 1, 1, 0, 0, 1]
    self.feature_favorited_row_lengths = [2, 1, 1, 2]
    # sparse tensor for friends:
    # row 0: 3
    # row 1: 0, 1, 2
    # row 2: 3
    # row 3: 0, 1, 2
    self.feature_friends_indices = [[0, 0], [1, 0], [1, 1], [1, 2],
                                    [2, 0], [3, 0], [3, 1], [3, 2]]
    self.feature_friends_values = [3, 0, 1, 2, 3, 0, 1, 2]
    self.feature_friends_row_lengths = [1, 3, 1, 3]

  def _create_mid_level(self):
    # [FUNCTION DOC]
    # HELPER: test utility. Creates the serving embedding object from the fixture feature
    # config and a plain SGD optimizer (learning rate 0.1). Returns it.
    optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    return tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=self.feature_config, optimizer=optimizer)

  def _get_dense_tensors(self, dtype=dtypes.int32):
    # [FUNCTION DOC]
    # HELPER: test utility. Builds three dense constant tensors from the watched, favorited and
    # friends value lists with the given element type (default 32-bit integer). Returns a tuple.
    feature0 = constant_op.constant(self.feature_watched_values, dtype=dtype)
    feature1 = constant_op.constant(self.feature_favorited_values, dtype=dtype)
    feature2 = constant_op.constant(self.feature_friends_values, dtype=dtype)
    return (feature0, feature1, feature2)

  def test_cpu_dense_lookup(self):
    # [FUNCTION DOC]
    # Scenario: dense id tensors, no weights. Action: call the embedding object. Expected: the
    # result equals a direct row gather from each table's numpy values for every feature.
    mid_level = self._create_mid_level()
    features = self._get_dense_tensors()
    results = mid_level(features, weights=None)
    all_lookups = []
    for feature, config in zip(nest.flatten(features), self.feature_config):
      table = mid_level.embedding_tables[config.table].numpy()
      all_lookups.append(table[feature.numpy()])
    self.assertAllClose(results, nest.pack_sequence_as(results, all_lookups))

  def test_cpu_dense_lookup_with_weights(self):
    # [FUNCTION DOC]
    # Scenario: dense ids with dense float weights. Expected: a value error saying a weight was
    # specified but the input is dense.
    mid_level = self._create_mid_level()
    features = self._get_dense_tensors()
    weights = self._get_dense_tensors(dtype=dtypes.float32)

    with self.assertRaisesRegex(
        ValueError, 'Weight specified for .*, but input is dense.'):
      mid_level(features, weights=weights)

  def _get_sparse_tensors(self, dtype=dtypes.int32):
    # [FUNCTION DOC]
    # HELPER: test utility. Builds three sparse tensors (watched, favorited, friends) from the
    # fixture indices and values with the given value type (default 32-bit integer) and dense
    # shapes of batch size four by two, two and three. Returns a tuple.
    feature0 = sparse_tensor.SparseTensor(
        indices=self.feature_watched_indices,
        values=constant_op.constant(self.feature_watched_values, dtype=dtype),
        dense_shape=[self.data_batch_size, 2])
    feature1 = sparse_tensor.SparseTensor(
        indices=self.feature_favorited_indices,
        values=constant_op.constant(self.feature_favorited_values, dtype=dtype),
        dense_shape=[self.data_batch_size, 2])
    feature2 = sparse_tensor.SparseTensor(
        indices=self.feature_friends_indices,
        values=constant_op.constant(self.feature_friends_values, dtype=dtype),
        dense_shape=[self.data_batch_size, 3])
    return (feature0, feature1, feature2)

  def test_cpu_sparse_lookup(self):
    # [FUNCTION DOC]
    # Scenario: sparse ids, no weights. Expected: per-row reduction of gathered table rows
    # (sum, or mean divided by row length for mean-combiner tables) matches the result.
    mid_level = self._create_mid_level()
    features = self._get_sparse_tensors()
    results = mid_level(features, weights=None)
    reduced = []
    for feature, config in zip(nest.flatten(features), self.feature_config):
      table = mid_level.embedding_tables[config.table].numpy()
      all_lookups = table[feature.values.numpy()]
      # With row starts we can use reduceat in numpy. Get row starts from the
      # ragged tensor API.
      ragged = ragged_tensor.RaggedTensor.from_sparse(feature)
      row_starts = ragged.row_starts().numpy()
      reduced.append(np.add.reduceat(all_lookups, row_starts))
      if config.table.combiner == 'mean':
        # for mean, divide by the row lengths.
        reduced[-1] /= np.expand_dims(ragged.row_lengths().numpy(), axis=1)
    self.assertAllClose(results, nest.pack_sequence_as(results, reduced))

  def test_cpu_sparse_lookup_with_weights(self):
    # [FUNCTION DOC]
    # Scenario: sparse ids with sparse float weights. Expected: gathered rows scaled by weights
    # and reduced per row; mean tables divide by the per-row weight sum. Compared to the result.
    mid_level = self._create_mid_level()
    features = self._get_sparse_tensors()
    weights = self._get_sparse_tensors(dtype=dtypes.float32)
    results = mid_level(features, weights=weights)
    weighted_sum = []
    for feature, weight, config in zip(nest.flatten(features),
                                       nest.flatten(weights),
                                       self.feature_config):
      table = mid_level.embedding_tables[config.table].numpy()
      # Expand dims here needed to broadcast this multiplication properly.
      weight = np.expand_dims(weight.values.numpy(), axis=1)
      all_lookups = table[feature.values.numpy()] * weight
      # With row starts we can use reduceat in numpy. Get row starts from the
      # ragged tensor API.
      row_starts = ragged_tensor.RaggedTensor.from_sparse(feature).row_starts()
      row_starts = row_starts.numpy()
      weighted_sum.append(np.add.reduceat(all_lookups, row_starts))
      if config.table.combiner == 'mean':
        weighted_sum[-1] /= np.add.reduceat(weight, row_starts)
    self.assertAllClose(results, nest.pack_sequence_as(results,
                                                       weighted_sum))

  def test_cpu_sparse_lookup_with_non_sparse_weights(self):
    # [FUNCTION DOC]
    # Scenario: sparse ids with dense weights. Expected: a value error stating the weight type
    # does not match the input type.
    mid_level = self._create_mid_level()
    features = self._get_sparse_tensors()
    weights = self._get_dense_tensors(dtype=dtypes.float32)
    with self.assertRaisesRegex(
        ValueError, 'but it does not match type of the input which is'):
      mid_level(features, weights=weights)

  def _get_ragged_tensors(self, dtype=dtypes.int32):
    # [FUNCTION DOC]
    # HELPER: test utility. Builds three ragged tensors from the fixture values and row lengths
    # with the given value type (default 32-bit integer). Returns a tuple.
    feature0 = ragged_tensor.RaggedTensor.from_row_lengths(
        values=constant_op.constant(self.feature_watched_values, dtype=dtype),
        row_lengths=self.feature_watched_row_lengths)
    feature1 = ragged_tensor.RaggedTensor.from_row_lengths(
        values=constant_op.constant(self.feature_favorited_values, dtype=dtype),
        row_lengths=self.feature_favorited_row_lengths)
    feature2 = ragged_tensor.RaggedTensor.from_row_lengths(
        values=constant_op.constant(self.feature_friends_values, dtype=dtype),
        row_lengths=self.feature_friends_row_lengths)
    return (feature0, feature1, feature2)

  def test_cpu_ragged_lookup_with_weights(self):
    # [FUNCTION DOC]
    # Scenario: ragged ids with ragged float weights. Expected: weighted per-row sums (divided
    # by weight sums for mean tables) computed in numpy match the result.
    mid_level = self._create_mid_level()
    features = self._get_ragged_tensors()
    weights = self._get_ragged_tensors(dtype=dtypes.float32)
    results = mid_level(features, weights=weights)
    weighted_sum = []
    for feature, weight, config in zip(nest.flatten(features),
                                       nest.flatten(weights),
                                       self.feature_config):
      table = mid_level.embedding_tables[config.table].numpy()
      # Expand dims here needed to broadcast this multiplication properly.
      weight = np.expand_dims(weight.values.numpy(), axis=1)
      all_lookups = table[feature.values.numpy()] * weight
      row_starts = feature.row_starts().numpy()
      weighted_sum.append(np.add.reduceat(all_lookups, row_starts))
      if config.table.combiner == 'mean':
        weighted_sum[-1] /= np.add.reduceat(weight, row_starts)
    self.assertAllClose(results, nest.pack_sequence_as(results,
                                                       weighted_sum))

  def test_cpu_partial_structure_for_features(self):
    # [FUNCTION DOC]
    # Scenario: only the first two features are passed (a subset of the feature config).
    # Expected: the lookup succeeds and matches the numpy reduction for those two features.
    mid_level = self._create_mid_level()
    # Remove one element of the tuple, the inputs are a subset of
    # feature_config and will be able to excute.
    features = tuple(self._get_sparse_tensors()[:2])
    results = mid_level(features, weights=None)
    reduced = []
    for i, feature in enumerate(nest.flatten(features)):
      config = self.feature_config[i]
      table = mid_level.embedding_tables[config.table].numpy()
      all_lookups = table[feature.values.numpy()]
      # With row starts we can use reduceat in numpy. Get row starts from the
      # ragged tensor API.
      ragged = ragged_tensor.RaggedTensor.from_sparse(feature)
      row_starts = ragged.row_starts().numpy()
      reduced.append(np.add.reduceat(all_lookups, row_starts))
      if config.table.combiner == 'mean':
        # for mean, divide by the row lengths.
        reduced[-1] /= np.expand_dims(ragged.row_lengths().numpy(), axis=1)
    self.assertAllClose(results, nest.pack_sequence_as(results, reduced))

  def test_cpu_invalid_structure_for_features(self):
    # [FUNCTION DOC]
    # Scenario: four feature inputs for a three-feature config. Expected: a value error.
    mid_level = self._create_mid_level()
    # Add one element of the tuple, self.feature_config has 3 so we need to
    # pass no more than 3 elements or None.
    features = self._get_sparse_tensors() + (self._get_sparse_tensors()[0],)
    with self.assertRaises(ValueError):
      mid_level(features, weights=None)

  def test_cpu_invalid_structure_for_weights(self):
    # [FUNCTION DOC]
    # Scenario: three sparse features but only two weights. Expected: a value error.
    mid_level = self._create_mid_level()
    features = self._get_sparse_tensors()
    # Remove one element of the tuple, inputs has 3 so we need to pass 3 (or
    # None).
    weights = tuple(self._get_dense_tensors(dtype=dtypes.float32)[:2])
    with self.assertRaises(ValueError):
      mid_level(features, weights=weights)

  def _numpy_sequence_lookup(self, table, indices, values, batch_size,
                             max_sequence_length, dim):
    # [FUNCTION DOC]
    # HELPER: numpy reference for sequence lookups. Inputs: table values, sparse indices and ids
    # (arrays), batch size, max sequence length and embedding dimension. Drops entries whose
    # position is at or beyond the max length, gathers rows, and scatters them into a zero array
    # of batch size by max length by dim. Returns that array.
    # First we truncate to max_sequence_length.
    valid_entries = np.nonzero(indices[:, 1] < max_sequence_length)[0]
    indices = indices[valid_entries]
    values = values[valid_entries]
    # Then we gather the values
    lookup = table[values]
    # Then we scatter them into the result array.
    scatter_result = np.zeros([batch_size, max_sequence_length, dim])
    for i, index in enumerate(indices):
      scatter_result[index[0], index[1], :] = lookup[i]
    return scatter_result

  def test_cpu_sequence_lookup_sparse(self):
    # [FUNCTION DOC]
    # Scenario: friends feature with max sequence length two, sparse input. Expected: result
    # equals the numpy sequence lookup golden array.
    feature_config = (
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_user, name='friends', max_sequence_length=2),)
    optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config, optimizer=optimizer)
    features = self._get_sparse_tensors()[2:3]
    result = mid_level(features, weights=None)

    golden = self._numpy_sequence_lookup(
        mid_level.embedding_tables[self.table_user].numpy(),
        features[0].indices.numpy(),
        features[0].values.numpy(),
        self.data_batch_size,
        feature_config[0].max_sequence_length,
        self.table_user.dim)

    self.assertAllClose(result[0], golden)

  def test_cpu_sequence_lookup_ragged(self):
    # [FUNCTION DOC]
    # Scenario: same as the sparse sequence test but with ragged input (converted to sparse for
    # the golden). Expected: result equals the numpy golden array.
    feature_config = (
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_user, name='friends', max_sequence_length=2),)
    optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config, optimizer=optimizer)
    features = self._get_ragged_tensors()[2:3]
    result = mid_level(features, weights=None)

    sparse_ver = features[0].to_sparse()
    golden = self._numpy_sequence_lookup(
        mid_level.embedding_tables[self.table_user].numpy(),
        sparse_ver.indices.numpy(),
        sparse_ver.values.numpy(),
        self.data_batch_size,
        feature_config[0].max_sequence_length,
        self.table_user.dim)

    self.assertAllClose(result[0], golden)

  def test_cpu_high_dimensional_lookup_ragged(self):
    # [FUNCTION DOC]
    # Scenario: friends feature with output shape two by two and ragged input of batch four.
    # Expected: the result shape is two by two by two (output shape plus embedding dim).
    feature_config = (tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_user, name='friends', output_shape=[2, 2]),)
    optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config, optimizer=optimizer)
    features = self._get_ragged_tensors()[2:3]
    result = mid_level(features, weights=None)

    self.assertAllClose(result[0].shape, (2, 2, 2))

  def test_cpu_high_dimensional_sequence_lookup_ragged(self):
    # [FUNCTION DOC]
    # Scenario: output shape two by four with batch four, so the element count is a multiple of
    # the batch and the quotient is the sequence length. Expected: result shape two, four, two.
    # Prod of output shape is a factor of the data batch size.
    # The divide result will be the sequence length.
    feature_config = (tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_user, name='friends', output_shape=[2, 4]),)
    optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config, optimizer=optimizer)
    features = self._get_ragged_tensors()[2:3]
    result = mid_level(features, weights=None)
    self.assertAllClose(result[0].shape, (2, 4, 2))

  def test_cpu_high_dimensional_invalid_lookup_ragged(self):
    # [FUNCTION DOC]
    # Scenario: output shape of three, which is not a factor of the batch size of four.
    # Expected: a value error saying the output shape should be the factor of the batch size.
    # Prod of output shape is not a factor of the data batch size.
    # An error will be raised in this case.
    feature_config = (tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_user, name='friends', output_shape=[3]),)
    optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config, optimizer=optimizer)
    features = self._get_ragged_tensors()[2:3]
    with self.assertRaisesRegex(
        ValueError,
        'Output shape set in the FeatureConfig should be the factor'):
      mid_level(features, weights=None)

  def test_cpu_no_optimizer(self):
    # [FUNCTION DOC]
    # Scenario: embedding created with no optimizer, then built manually. Expected: the table's
    # variables contain only the parameters entry (no slot variables).
    feature_config = (
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='watched', max_sequence_length=2),)
    mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config, optimizer=None)
    # Build the layer manually to create the variables. Normally calling enqueue
    # would do this.
    mid_level.build()
    self.assertEqual(
        list(mid_level._variables[self.table_video.name].keys()),
        ['parameters'])

  def test_cpu_multiple_creation(self):
    # [FUNCTION DOC]
    # Scenario: two embedding objects share one feature config and optimizer. Expected: both
    # build on CPU without error (no assertion beyond not raising).
    feature_config = (tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_user, name='friends', max_sequence_length=2),)
    optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    embedding_one = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config, optimizer=optimizer)
    embedding_two = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config, optimizer=optimizer)

    # Both of the tpu embedding tables should be able to build on cpu.
    embedding_one.build()
    embedding_two.build()


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
