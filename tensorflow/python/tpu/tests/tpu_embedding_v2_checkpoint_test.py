# [MODULE DOC]
# Test module for checkpointing of the V2 TPU embedding mid-level API. Covers saving a model whose
# embedding values live on the TPU, restoring into a new embedding (including before variables are
# created), exporting to a CPU model for serving, and name parity between TPU and CPU checkpoints.
# Needs a TPU; runs with v2 behavior via the test main entry. Builds on the shared base test module.
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
import numpy as np
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.tpu import tpu_embedding_for_serving
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test
from tensorflow.python.training import checkpoint_utils


class TPUEmbeddingCheckpointTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):
  # [CLASS DOC]
  # Test case for V2 embedding checkpoint behavior, derived from the shared TPU embedding base test.
  # Adds a helper to save a checkpoint and fetch the single table variable from it. Uses base fixtures
  # (strategy creation, resolver, temp directory helper, skip helper).

  def make_checkpoint_and_get_embedding(self, name, model, num_rows):
    # [FUNCTION DOC]
    # HELPER: saves a model checkpoint and reads back the embedding table values.
    # Inputs: name, a sub-directory name under the temp dir; model, the object to checkpoint; num_rows,
    # number of table rows expected. Saves a checkpoint under the name, lists variables in it, keeps those
    # whose shape is num_rows by 4 (found by shape because the key can change), and loads that variable.
    # Raises a runtime error unless exactly one such variable exists. Returns the loaded array.
    # KNOWN ISSUE: the in-code comment mentions a shape of 8 by 4 while the shape actually matched is
    # rows-by-4 using the given row count.
    """Saves model to checkpoint name, retrieves embedding variables."""
    checkpoint = util.Checkpoint(model=model)
    checkpoint.save(self._get_tmpdir(name, 'save'))

    # Get the name of the table video variable which should be the only
    # [8, 4] shaped tensor in the checkpoint. Note that we do this
    # as the key can change.
    variables = checkpoint_utils.list_variables(self._get_tmpdir(name))
    variables = [name for name, size in variables if size == [num_rows, 4]]
    if len(variables) != 1:
      raise RuntimeError('Found {} copies of the parameter variable in the '
                         'checkpoint. Exactly one copy exported.'.format(
                             len(variables)))
    return checkpoint_utils.load_variable(self._get_tmpdir(name), variables[0])

  def test_checkpoint_save_retrieves(self):
    # [FUNCTION DOC]
    # Scenario: saving a checkpoint retrieves the current TPU values. Builds an embedding with one table
    # (rows equal to replica count, dim 4, sum combiner) initialized to ones, loads its variables to the
    # TPU, and checks a saved checkpoint holds ones. Reinitializes the TPU, builds a second embedding
    # initialized to twos and loads it to the TPU. Then checkpointing the first embedding must now yield
    # twos because the checkpoint retrieves from the TPU. Asserts both with explanatory messages.
    strategy = self._get_strategy()
    num_rows = strategy.num_replicas_in_sync

    with strategy.scope():
      first_mid_level_contents = np.ones((num_rows, 4))
      first_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      initializer = init_ops_v2.Constant(first_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=num_rows,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)

      first_mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config, first_mid_level_optimizer)
      first_mid_level.build(64)

    # Ensure that the variables from the first model are loaded.
    first_mid_level._load_variables()

    self.assertAllClose(
        first_mid_level_contents,
        self.make_checkpoint_and_get_embedding('before_load', first_mid_level,
                                               num_rows),
        msg='Checkpoint should contain values from the first api object.')

    # Reinitialize the tpu.
    tpu_cluster_resolver.initialize_tpu_system(self.resolver)

    with strategy.scope():
      second_mid_level_contents = np.ones((num_rows, 4)) * 2
      second_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      initializer = init_ops_v2.Constant(second_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=num_rows,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)
      second_mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config, second_mid_level_optimizer)
      second_mid_level.build(64)

    second_mid_level._load_variables()

    # When we load the variables from the second mid level API object to the TPU
    # we expect that checkpointing the first mid level API object will now
    # retrieve the values from the TPU which are now different from the current
    # variables in the first mid level.
    self.assertAllClose(
        second_mid_level_contents,
        self.make_checkpoint_and_get_embedding('after_load', first_mid_level,
                                               num_rows),
        msg='Checkpoint should contain values from the second api object.')

  def test_checkpoint_restore_loads(self):
    # [FUNCTION DOC]
    # Scenario: restoring a checkpoint loads values onto the TPU. Builds an embedding initialized to ones,
    # loads and checkpoints it, reinitializes the TPU, builds a second embedding initialized to twos and
    # loads it. Asserts the second embedding holds its initial twos, restores the first checkpoint into
    # it, retrieves variables from the TPU, and asserts the second embedding now holds the ones.
    strategy = self._get_strategy()
    num_rows = strategy.num_replicas_in_sync

    def get_values(mid):
      # [FUNCTION DOC]
      # HELPER: nested accessor. Input: mid, a TPU embedding. Returns the first underlying variable of the
      # table parameters converted to a tensor.
      return ops.convert_to_tensor(
          mid._variables['table']['parameters'].variables[0])

    with strategy.scope():
      first_mid_level_contents = np.ones((num_rows, 4))
      first_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      initializer = init_ops_v2.Constant(first_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=num_rows,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)

      first_mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config, first_mid_level_optimizer)
      first_mid_level.build(64)

    first_mid_level._load_variables()

    first_checkpoint = util.Checkpoint(model=first_mid_level)
    first_checkpoint.save(self._get_tmpdir('restore', 'save'))

    tpu_cluster_resolver.initialize_tpu_system(self.resolver)

    with strategy.scope():
      second_mid_level_contents = np.ones((num_rows, 4)) * 2
      second_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      initializer = init_ops_v2.Constant(second_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=num_rows,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)
      second_mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config, second_mid_level_optimizer)
      second_mid_level.build(64)

    second_mid_level._load_variables()

    self.assertAllClose(
        second_mid_level_contents,
        get_values(second_mid_level),
        msg='Second mid level api should contain its initial values.',
    )
    # We restore the checkpoint of our first model into our second model.
    # This should load the first mid level API object onto the TPU.
    second_checkpoint = util.Checkpoint(model=second_mid_level)
    second_checkpoint.restore(self._get_tmpdir('restore', 'save-1'))

    # Call retrieve here as a way to check what the TPU contains.
    # Calling the retrieve ops directly might make for a cleaner separation of
    # test and module, though.
    second_mid_level._retrieve_variables()

    self.assertAllClose(
        first_mid_level_contents,
        get_values(second_mid_level),
        msg='Second mid level api should have retrieved the first model values.'
    )

  def test_checkpoint_restore_before_variable_creation(self):
    # [FUNCTION DOC]
    # Scenario: restore into a module whose embedding variables are not yet created. Skipped when the
    # project or zone flag is set. Builds a module with an embedding initialized to ones (twice the
    # replica count rows), creates the embedding, and saves a checkpoint. Reinitializes the TPU and
    # creates a second module initialized to zeros, restores the checkpoint before creating its embedding,
    # then creates the embedding. Asserts the values are ones, retrieves variables from the TPU, and
    # asserts they are still ones.
    self.skip_if_oss()

    class TestModule(module.Module):
      # [CLASS DOC]
      # HELPER: small module used by the restore-before-creation test.
      # Module holding the initializer, the row count and a TPU embedding (one table, one feature, plain
      # gradient descent) created in the constructor; the embedding itself is built later on request.

      def __init__(self, initializer, rows):
        # [FUNCTION DOC]
        # Constructor. Inputs: initializer for table values; rows, vocabulary size. Stores both, creates a
        # one-table configuration (dim 4, sum combiner), one feature, a default gradient descent optimizer,
        # and a TPU embedding object stored as an attribute. Returns nothing.
        self._initializer = initializer
        self._rows = rows

        table = tpu_embedding_v2_utils.TableConfig(
            vocabulary_size=self._rows,
            dim=4,
            initializer=self._initializer,
            combiner='sum',
            name='table')
        feature_config = (tpu_embedding_v2_utils.FeatureConfig(
            table=table, name='feature'),)
        optimizer = tpu_embedding_v2_utils.SGD()

        self.tpu_embedding = tpu_embedding_v2.TPUEmbedding(
            feature_config, optimizer)

      def create_embedding(self):
        # [FUNCTION DOC]
        # Builds the embedding with a nominal batch size of 64 (irrelevant because no training happens),
        # which creates its variables. Returns nothing.
        # We aren't training so batch_size here doesn't matter.
        self.tpu_embedding.build(64)

    strategy = self._get_strategy()
    with strategy.scope():
      module1 = TestModule(init_ops_v2.Ones(),
                           strategy.num_replicas_in_sync * 2)
      module1.create_embedding()

    checkpoint = util.Checkpoint(test_module=module1)
    checkpoint.save(self._get_tmpdir('restore_before_create', 'save'))

    # Reinitialize the tpu
    strategy = self._get_strategy()

    with strategy.scope():
      module2 = TestModule(init_ops_v2.Zeros(),
                           strategy.num_replicas_in_sync * 2)

    checkpoint = util.Checkpoint(test_module=module2)
    checkpoint.restore(self._get_tmpdir('restore_before_create', 'save-1'))

    with strategy.scope():
      module2.create_embedding()

    def get_values(mid):
      # [FUNCTION DOC]
      # HELPER: nested accessor. Input: mid, a TPU embedding. Returns the numpy array of the first
      # underlying variable of the table parameters.
      return mid._variables['table']['parameters'].variables[0].numpy()

    self.assertAllClose(
        np.ones((strategy.num_replicas_in_sync * 2, 4)),
        get_values(module2.tpu_embedding))

    # Fetch the values from the TPU to check that they are the same.
    module2.tpu_embedding._retrieve_variables()

    self.assertAllClose(
        np.ones((strategy.num_replicas_in_sync * 2, 4)),
        get_values(module2.tpu_embedding))

  def test_model_export_cpu(self):
    # [FUNCTION DOC]
    # Scenario: export a TPU-trained embedding to a CPU serving model. Builds a TPU embedding (ones) and a
    # second embedding for the CPU with the same config, loads the first to the TPU, saves its checkpoint
    # and restores it into the CPU one. Defines a serving function that does a CPU embedding lookup, saves
    # the CPU model with a signature for an int32 vector of length two, loads it back and predicts for
    # ids one and zero. Asserts the prediction equals the matching rows of the original contents.
    strategy = self._get_strategy()
    num_rows = strategy.num_replicas_in_sync

    with strategy.scope():
      first_mid_level_contents = np.ones((num_rows, 4))
      first_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      initializer = init_ops_v2.Constant(first_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=num_rows,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)

      first_mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config, first_mid_level_optimizer)

      first_mid_level.build(64)

    cpu_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    cpu_mid_level = tpu_embedding_v2.TPUEmbedding(feature_config,
                                                  cpu_mid_level_optimizer)

    cpu_mid_level.build(64)

    first_mid_level._load_variables()

    tpu_checkpoint = util.Checkpoint(model=first_mid_level)
    tpu_checkpoint.save(self._get_tmpdir('export_cpu', 'save'))

    # We restore the checkpoint of our tpu mid level onto our cpu mid level.
    cpu_checkpoint = util.Checkpoint(model=cpu_mid_level)
    cpu_checkpoint.restore(self._get_tmpdir('export_cpu', 'save-1'))

    @def_function.function
    def serve_tensors(features):
      # [FUNCTION DOC]
      # HELPER: serving function traced into the exported signature. Input: features. Performs a CPU
      # embedding lookup over the CPU tables and returns the first feature's activations.
      features = tpu_embedding_for_serving.cpu_embedding_lookup(
          features, None, cpu_mid_level.embedding_tables,
          cpu_mid_level._feature_config)
      return features[0]

    signatures = {
        'serving_default':
            serve_tensors.get_concrete_function((tensor_spec.TensorSpec(
                shape=(2,), dtype=dtypes.int32, name='feature'),))
    }
    save.save(
        cpu_mid_level,
        export_dir=self._get_tmpdir('export_cpu', 'exported_model'),
        signatures=signatures)

    imported = load.load(self._get_tmpdir('export_cpu', 'exported_model'))
    predict_fn = imported.signatures['serving_default']

    input_feature_value = np.array([1, 0])
    input_batch = (constant_op.constant(
        input_feature_value, dtype=dtypes.int32),)
    prediction = predict_fn(*input_batch)['output_0']
    self.assertAllClose(prediction.numpy(),
                        first_mid_level_contents[input_feature_value])

  @parameterized.parameters(tpu_embedding_v2_utils.SGD,
                            tpu_embedding_v2_utils.Adagrad,
                            tpu_embedding_v2_utils.Adam,
                            tpu_embedding_v2_utils.FTRL)
  def test_check_checkpoint_variable_names_are_same_on_cpu_and_tpu(
      self, optimizer):
    # [FUNCTION DOC]
    # Parameterized over optimizer classes (gradient descent, adagrad, adam, ftrl). Input: optimizer
    # class. Non-gradient-descent classes are skipped when the project or zone flag is set. Builds two
    # embeddings with the same config and optimizer, saves each into its own temp directory, lists each
    # checkpoint's variables, and asserts the lists are equal.
    # Reinitialize the TPU so that we can re-initialize the embeddings with the
    # given optimizer.
    if optimizer != tpu_embedding_v2_utils.SGD:
      self.skip_if_oss()
    strategy = self._get_strategy()
    num_rows = strategy.num_replicas_in_sync

    with strategy.scope():
      first_mid_level_contents = np.ones((num_rows, 4))
      first_mid_level_optimizer = optimizer(learning_rate=0.1)
      initializer = init_ops_v2.Constant(first_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=num_rows,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)

      first_mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config, first_mid_level_optimizer)

      first_mid_level.build(64)

    cpu_mid_level_optimizer = optimizer(learning_rate=0.1)
    cpu_mid_level = tpu_embedding_v2.TPUEmbedding(feature_config,
                                                  cpu_mid_level_optimizer)
    cpu_mid_level.build(64)

    tpu_checkpoint = util.Checkpoint(model=first_mid_level)
    tpu_checkpoint.save(self._get_tmpdir('save-tpu', 'save'))
    tpu_variables = checkpoint_utils.list_variables(
        self._get_tmpdir('save-tpu'))

    cpu_checkpoint = util.Checkpoint(model=cpu_mid_level)
    cpu_checkpoint.save(self._get_tmpdir('save-cpu', 'save'))
    cpu_variables = checkpoint_utils.list_variables(
        self._get_tmpdir('save-cpu'))

    self.assertAllEqual(tpu_variables, cpu_variables)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
