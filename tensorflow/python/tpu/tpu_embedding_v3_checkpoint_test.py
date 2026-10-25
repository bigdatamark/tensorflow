# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
# Test module for checkpointing of the sharded TPU embedding model defined in
# tpu_embedding_v3. Defines one test class that saves a model, restores into a second model
# and checks the values come from the checkpoint. Needs a TPU; when run as a script it
# enables v2 behavior and the MLIR bridge before starting the test runner.
"""Tests for tpu_embedding_v3_checkpoint."""
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.framework import config
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_embedding_v3


class TPUEmbeddingV3CheckpointTest(parameterized.TestCase, test.TestCase):
  # [CLASS DOC]
  # Test case covering checkpoint save and restore of the sharded TPU embedding model
  # (TPUEmbeddingV2 from tpu_embedding_v3). Shared fixtures are the vocabulary size and
  # embedding dimension set in setUp, plus a helper that builds temporary checkpoint
  # directory paths. Requires a real TPU system.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture run before each test. Calls the parent setUp and stores the vocabulary
    # size (16384) and embedding dimension (128) on the instance. Returns nothing.
    super().setUp()
    self.vocabulary_size = 16384
    self.embedding_dim = 128

  def test_checkpoint_save_and_restore(self):
    # [FUNCTION DOC]
    # Scenario: two feature configurations differ only in the table initializer (constant 1
    # versus constant 2). Initializes the TPU system and a TPU strategy, builds a model with
    # the first configuration inside the strategy scope and saves a checkpoint from inside
    # the scope. Asserts the created variable is a sharded embedding variable with one shard
    # per replica, each shard having the expected shape (vocabulary divided by replicas, by
    # dimension) and filled with ones. Then creates a second model with the other
    # initializer, replaces its table-initialization routine with one that fails the test if
    # called, restores the saved checkpoint, builds it, and asserts the restored shard
    # values are ones (so the checkpoint values, not the initializer, were used).
    # Side effects: TPU system initialization and writing checkpoint files under the
    # temporary directory. Returns nothing.
    feature_config_1 = (
        tpu_embedding_v2_utils.FeatureConfig(
            table=tpu_embedding_v2_utils.TableConfig(
                vocabulary_size=self.vocabulary_size,
                dim=self.embedding_dim,
                initializer=init_ops_v2.Constant(1.0),
                optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1),
                combiner="sum",
                name="video"),
            name="watched",
            output_shape=[16]))

    feature_config_2 = (
        tpu_embedding_v2_utils.FeatureConfig(
            table=tpu_embedding_v2_utils.TableConfig(
                vocabulary_size=self.vocabulary_size,
                dim=self.embedding_dim,
                initializer=init_ops_v2.Constant(2.0),  # different initializer
                optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1),
                combiner="sum",
                name="video"),
            name="watched",
            output_shape=[16]))

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu="")
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    with strategy.scope():
      model1 = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config_1,
          optimizer=tpu_embedding_v2_utils.SGD())
      model1.build()

      # Check saving from inside scope works.
      checkpoint = util.Checkpoint(model=model1)
      checkpoint.save(self._get_tmpdir("restore", "save"))

    # Check the variable created by model1
    expected_shard_shape = (self.vocabulary_size //
                            strategy.num_replicas_in_sync, self.embedding_dim)
    self.assertIsInstance(model1._variables["video"]["parameters"],
                          tpu_embedding_v3.TPUEmbeddingShardedVariable)
    self.assertLen(model1._variables["video"]["parameters"].values,
                   strategy.num_replicas_in_sync)
    self.assertEqual(model1._variables["video"]["parameters"].values[0].shape,
                     expected_shard_shape)
    self.assertAllEqual(
        model1._variables["video"]["parameters"].values[0].numpy(),
        np.ones(expected_shard_shape) * 1.0)

    with strategy.scope():
      model2 = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config_2,
          optimizer=tpu_embedding_v2_utils.SGD())

      def fail_initializer(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: stand-in for the model's table initializer in the restore test.
        # Accepts and discards any arguments, then fails the test with a message saying the
        # initializer must not run when restoring from a checkpoint. Returns nothing.
        del args, kwargs
        self.fail("initializer should not be called when restoring")

      assert model2._batch_initialize_tables
      model2._batch_initialize_tables = fail_initializer

    checkpoint = util.Checkpoint(model=model2)

    # Load from checkpoint
    checkpoint.restore(self._get_tmpdir("restore", "save-1"))
    model2.build()

    # Check the variable restored by model2
    self.assertAllEqual(
        model2._variables["video"]["parameters"].values[0].numpy(),
        np.ones(expected_shard_shape) * 1.0)

  def _get_tmpdir(self, name, subdir=""):
    # [FUNCTION DOC]
    # HELPER: builds checkpoint paths for the tests in this class.
    # Inputs: name, a directory name, and subdir, an optional sub-name (default empty).
    # Returns a path joining the test temporary directory (from the environment, falling
    # back to the system temp directory), name, and subdir when it is non-empty. Creates
    # nothing on disk.
    segments = [os.environ.get("TEST_TMPDIR", "/tmp"), name] + (
        [subdir] if subdir else []
    )
    return os.path.join(*segments)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  config.enable_mlir_bridge()
  test.main()
