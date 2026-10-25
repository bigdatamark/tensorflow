# [MODULE DOC]
# TPU test module for initialization behavior of the mid level TPU embedding API (version 2):
# using it on CPU, creating several embedding objects, and determinism of the generated
# configuration. Defines one test class on top of the shared embedding base test; runs under the
# TF test runner with V2 behavior when executed as a script. No public API, no import-time side
# effects beyond defining the class.
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
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test


class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):
  # [CLASS DOC]
  # Test case for initialization and configuration of the TPU embedding mid level API. Uses the
  # strategy, table and feature fixtures provided by the shared base test class.

  def test_enqueue_dequeue_apply_gradients_on_cpu(self):
    # [FUNCTION DOC]
    # Test: creates a mid level embedding object outside any TPU strategy and asserts that
    # dequeue, enqueue (with a small sparse watched feature) and apply gradients (with nothing as
    # gradients, on a freshly created object) each raise a runtime error, since those operations
    # are only supported on TPU.
    # Dequeue on CPU.
    mid_level_api = self._create_mid_level()
    with self.assertRaises(RuntimeError):
      mid_level_api.dequeue()
    # Enqueue on CPU.
    features = {
        'watched': sparse_tensor.SparseTensor(
            indices=self.feature_watched_indices,
            values=self.feature_watched_values,
            dense_shape=[2, 2])}
    with self.assertRaises(RuntimeError):
      mid_level_api.enqueue(features)
    # Apply gradient on CPU.
    mid_level_api = self._create_mid_level()
    with self.assertRaises(RuntimeError):
      mid_level_api.apply_gradients(None)

  def test_multiple_creation(self):
    # [FUNCTION DOC]
    # Test: under one strategy scope creates two embedding objects sharing the same feature
    # configuration (sequence length two) and SGD optimizer. Building the first succeeds; building
    # the second must raise a runtime error saying TPU is already initialized for embeddings.
    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_user, name='friends', max_sequence_length=2)
    optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    strategy = self._get_strategy()
    with strategy.scope():
      embedding_one = tpu_embedding_v2.TPUEmbedding(
          feature_config=feature_config, optimizer=optimizer)
      embedding_two = tpu_embedding_v2.TPUEmbedding(
          feature_config=feature_config, optimizer=optimizer)

    # The first TPU embedding should be able to be built.
    # The second one should fail with a runtime error indicating another TPU
    # embedding has already been initialized on TPU.
    embedding_one.build(64)
    with self.assertRaisesRegex(RuntimeError,
                                'TPU is already initialized for embeddings.'):
      embedding_two.build(64)

  def test_same_config_different_instantiations(self):
    # [FUNCTION DOC]
    # Test: draws random dimensions and vocabulary sizes for thirty tables once, then builds the
    # embedding configuration twice from the same data and asserts the two generated protocol
    # buffers are equal, i.e. config generation is deterministic (including table order and the
    # per-table Adagrad optimizers used for even dimensions).
    num_tables = 30
    table_dim = np.random.randint(1, 128, size=[num_tables])
    table_vocab_size = np.random.randint(100, 1000, size=[num_tables])
    table_names = ['table{}'.format(i) for i in range(num_tables)]
    table_data = list(zip(table_dim, table_vocab_size, table_names))
    strategy = self._get_strategy()

    def tpu_embedding_config():
      # [FUNCTION DOC]
      # Builds one feature config per entry of the shared table data (optional Adagrad optimizer
      # with a callable learning rate for even dimensions, zero initializer), creates the mid
      # level embedding under the strategy scope with an Adagrad default optimizer, sets the
      # output shapes to a vector of 128 per feature and returns the generated configuration
      # protocol buffer. Takes no inputs.
      feature_configs = []
      for dim, vocab, name in table_data:
        optimizer = None
        if dim % 2 == 0:
          optimizer = tpu_embedding_v2_utils.Adagrad(
              learning_rate=lambda: constant_op.constant(1.0))

        feature_configs.append(
            tpu_embedding_v2_utils.FeatureConfig(
                table=tpu_embedding_v2_utils.TableConfig(
                    vocabulary_size=int(vocab),
                    dim=int(dim),
                    initializer=init_ops_v2.Zeros(),
                    optimizer=optimizer,
                    name=name)))
      optimizer = tpu_embedding_v2_utils.Adagrad(learning_rate=0.1)
      with strategy.scope():
        mid_level_api = tpu_embedding_v2.TPUEmbedding(
            feature_config=feature_configs, optimizer=optimizer)
      mid_level_api._output_shapes = [TensorShape(128)] * len(feature_configs)
      return mid_level_api._create_config_proto()

    self.assertProtoEquals(tpu_embedding_config(), tpu_embedding_config())

  def test_learning_rate_tag_order(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the feature config append sits outside the loop (wrong indentation), so only
    # one feature config is created (with the last optimizer) although thirty tables are
    # suggested; the order check therefore only covers a single table.
    # Test: builds the embedding configuration and asserts that each table descriptor's dynamic
    # learning rate tag equals its index in the descriptor list.
    num_tables = 30
    strategy = self._get_strategy()

    feature_configs = []
    for i in range(num_tables):
      optimizer = tpu_embedding_v2_utils.Adagrad(
          learning_rate=lambda: constant_op.constant(1.0))

    feature_configs.append(
        tpu_embedding_v2_utils.FeatureConfig(
            table=tpu_embedding_v2_utils.TableConfig(
                vocabulary_size=100,
                dim=128,
                initializer=init_ops_v2.Zeros(),
                optimizer=optimizer)))
    with strategy.scope():
      mid_level_api = tpu_embedding_v2.TPUEmbedding(
          feature_config=feature_configs, optimizer=optimizer)
    mid_level_api._output_shapes = [TensorShape(128)] * len(feature_configs)
    result = mid_level_api._create_config_proto()
    for i, table in enumerate(result.table_descriptor):
      self.assertEqual(i,
                       table.optimization_parameters.learning_rate.dynamic.tag)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
