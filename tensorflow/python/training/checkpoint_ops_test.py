# [MODULE DOC]
# Functional tests for the private checkpoint remapping wrappers in the checkpoint_ops module.
# Contains one graph-mode test case (V1 name-based checkpoints only) that saves a small
# checkpoint plus vocabulary files and verifies matrix and embedding loading with row/column
# remapping, out-of-vocabulary buckets, partitioned variables and default initializers.
# Running the file executes the test main entry point.
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for Python wrappers around warm-starting."""
import os

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_ops
from tensorflow.python.training import saver as saver_lib


@test_util.run_v1_only(
    'This is to test V1 name-based checkpoints which is not supported in V2.')
class LoadAndRemapWrappersTest(test.TestCase):
  # [CLASS DOC]
  # Test case (graph mode, V1 name-based checkpoints only) for the private checkpoint remapping
  # wrappers: the matrix loader, the matrix initializer factory and the embedding initializer.
  # Shared fixture, created in setUp: a saved checkpoint with a 5 by 16 embeddings variable
  # holding consecutive values, old/new feature vocab files, old/new class vocab files, and an
  # initializer that fills missing entries with the constant 42.
  """Tests for the functionality of the Python wrappers."""

  def setUp(self):
    # [FUNCTION DOC]
    # Builds the fixture for every test. Resets the default graph, saves a checkpoint (step 5)
    # in the temporary directory containing a variable named some_scope/embeddings of shape 5 by
    # 16 with values 0 to 79 in row-major order, and stores its path in the checkpoint file
    # attribute. Writes four vocabulary text files (new feature: zero..four; old feature:
    # zero..three; new class: MISSING, knitting, flask, eminem; old class: knitting, eminem,
    # MISSING), stores their paths, and defines a constant-value initializer (42) as an attribute.
    # Takes no inputs and returns nothing; side effects are files on disk and graph state.
    ops.reset_default_graph()
    # Create the checkpoint file in a temporary directory.
    checkpoint_prefix = os.path.join(self.get_temp_dir(), 'model')
    # 0., 1., ..., 79. reshaped into [5, 16].
    initializer = init_ops.constant_initializer(
        np.reshape(np.linspace(0.0, 79, 5 * 16), (5, 16)))
    with self.cached_session() as sess:
      with variable_scope.variable_scope('some_scope'):
        variable_scope.get_variable(name='embeddings', shape=[5, 16],
                                    initializer=initializer)
      self.evaluate(variables.global_variables_initializer())
      saver = saver_lib.Saver()
      saver.save(sess, checkpoint_prefix, global_step=5)
    self.checkpoint_file = '{}-5'.format(checkpoint_prefix)

    # Create the vocabulary files.
    self.new_feature_vocab_file = os.path.join(
        self.get_temp_dir(), 'new_feature_vocab.txt')
    with open(self.new_feature_vocab_file, 'w') as f:
      f.write('\n'.join(['zero', 'one', 'two', 'three', 'four']) + '\n')

    self.old_feature_vocab_file = os.path.join(
        self.get_temp_dir(), 'old_feature_vocab.txt')
    with open(self.old_feature_vocab_file, 'w') as f:
      f.write('\n'.join(['zero', 'one', 'two', 'three']) + '\n')

    self.new_class_vocab_file = os.path.join(
        self.get_temp_dir(), 'new_class_vocab.txt')
    with open(self.new_class_vocab_file, 'w') as f:
      f.write('\n'.join(['MISSING', 'knitting', 'flask', 'eminem']) + '\n')

    self.old_class_vocab_file = os.path.join(
        self.get_temp_dir(), 'old_class_vocab.txt')
    with open(self.old_class_vocab_file, 'w') as f:
      f.write('\n'.join(['knitting', 'eminem', 'MISSING']) + '\n')

    self.init_val = 42

    def _init_val_initializer(shape, dtype=None, partition_info=None):
      # [FUNCTION DOC]
      # HELPER: test initializer that supplies the recognizable constant fill value.
      # Ignores dtype and partition_info and returns a float32 tensor of the requested shape in
      # which every element equals the fixture's constant (42), so tests can tell newly
      # initialized entries from entries loaded from the checkpoint.
      del dtype, partition_info  # Unused by this unit-testing initializer.
      return array_ops.tile(
          constant_op.constant([[self.init_val]], dtype=dtypes.float32), shape)

    self.initializer = _init_val_initializer

  def test_load_and_remap_matrix(self):
    # [FUNCTION DOC]
    # Scenario: remap both rows (feature vocab) and columns (class vocab) with one row offset,
    # loading four rows and four columns plus one OOV row and one OOV column. Action: call the
    # private matrix loader and evaluate the result. Expected: a 5 by 5 matrix whose found
    # entries match the checkpoint values at the remapped positions and whose missing and OOV
    # entries equal the constant fill value.
    """Tests the end-to-end loading / remapping of weights."""
    # _load_and_remap_matrix() is the generalized wrapper that takes in row and
    # column vocabulary files, calls the relevant remappings, and returns the
    # weight matrix.  Take this example to be linear multi-class by providing
    # both row and column vocabularies.
    remapped_matrix = checkpoint_ops._load_and_remap_matrix(
        new_row_vocab_file=self.new_feature_vocab_file,
        old_row_vocab_file=self.old_feature_vocab_file,
        num_rows_to_load=4,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.checkpoint_file],
        new_row_vocab_offset=1,
        initializer=self.initializer,
        num_row_oov_buckets=1,
        num_col_oov_buckets=1)

    # [4 in vocab + 1 oov features, 4 in vocab + 1 oov classes].  The offset
    # means we read from the first line.
    expected_remapped_matrix = np.concatenate(
        [
            np.reshape([18, 34, 50, self.init_val, self.init_val], [5, 1]),
            np.reshape([16, 32, 48, self.init_val, self.init_val], [5, 1]),
            np.reshape([self.init_val] * 5, [5, 1]),
            np.reshape([17, 33, 49, self.init_val, self.init_val], [5, 1]),
            np.reshape([self.init_val] * 5, [5, 1])
        ],
        axis=1)

    with self.cached_session():
      self.assertAllClose(expected_remapped_matrix,
                          self.evaluate(remapped_matrix))

  def test_load_and_remap_output_layer_weight_initializer_linear(self):
    # [FUNCTION DOC]
    # Scenario: linear multi-class output layer; both feature (row) and class (column) vocabs
    # are remapped with one row OOV bucket and one column OOV bucket. Action: build the matrix
    # initializer, use it for a variable of shape 6 by 5 split into two row partitions, run the
    # variable initializers. Expected: the assembled variable equals the remapped checkpoint
    # values, with the constant fill for the missing feature row and OOV row and column,
    # confirming the partition offset logic.
    """Tests for the output layer initializer in the linear multi-class case."""
    loading_initializer = (checkpoint_ops._load_and_remap_matrix_initializer(
        new_row_vocab_size=5,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.checkpoint_file],
        new_row_vocab_file=self.new_feature_vocab_file,
        old_row_vocab_file=self.old_feature_vocab_file,
        num_row_oov_buckets=1,
        num_col_oov_buckets=1,
        initializer=self.initializer))

    # The new weight matrix is of size
    # [5 feature vocab + 1 feature OOV, 4 class vocab + 1 class OOV].  Use a
    # partitioned variable to confirm that the offset logic works.
    expected_remapped_matrix = np.concatenate(
        [
            np.reshape([2, 18, 34, 50, self.init_val, self.init_val], [6, 1]),
            np.reshape([0, 16, 32, 48, self.init_val, self.init_val], [6, 1]),
            np.reshape([self.init_val] * 6, [6, 1]),
            np.reshape([1, 17, 33, 49, self.init_val, self.init_val], [6, 1]),
            np.reshape([self.init_val] * 6, [6, 1])
        ],
        axis=1)
    remapped_matrix = variable_scope.get_variable(
        name='linear/obtained_weight_matrix',
        shape=[6, 5],
        initializer=loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(expected_remapped_matrix, remapped_matrix.as_tensor())

  def test_load_and_remap_output_layer_weight_initializer_dnn_output(self):
    # [FUNCTION DOC]
    # Scenario: DNN output layer where only the class (column) vocab is remapped and there is
    # one column OOV bucket; rows are taken directly from the checkpoint (five rows, no row
    # vocab files). Action: build the initializer and create a 5 by 5 variable in two row
    # partitions. Expected: columns follow the class remapping, the missing class column and the
    # OOV column hold the constant fill value, and all five rows come from the checkpoint.
    """Tests for the output layer initializer in the DNN output case."""
    loading_initializer = (checkpoint_ops._load_and_remap_matrix_initializer(
        new_row_vocab_size=5,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.checkpoint_file],
        num_col_oov_buckets=1,
        initializer=self.initializer))

    # The new weight matrix is of size
    # [5-sized input layer, 4 class vocab + 1 class OOV].
    expected_remapped_matrix = np.concatenate(
        [
            np.reshape([2, 18, 34, 50, 66], [5, 1]),
            np.reshape([0, 16, 32, 48, 64], [5, 1]),
            np.reshape([self.init_val] * 5, [5, 1]),
            np.reshape([1, 17, 33, 49, 65], [5, 1]),
            np.reshape([self.init_val] * 5, [5, 1])
        ],
        axis=1)
    remapped_matrix = variable_scope.get_variable(
        name='dnn_output/obtained_weight_matrix',
        shape=[5, 5],
        initializer=loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(expected_remapped_matrix, remapped_matrix.as_tensor())

  def test_initializer_with_oov_only_partition(self):
    # [FUNCTION DOC]
    # Scenario: five row OOV buckets and one column OOV bucket so that the second of two row
    # partitions contains only OOV rows. Action: build the initializer and create a 10 by 5
    # variable with two row partitions. Expected: the first partition holds remapped checkpoint
    # rows, then constant fill rows, and the OOV-only partition is entirely the fill value,
    # exercising the early-return path of the initializer.
    """Tests for the output layer initializer where one partition is all OOV."""
    loading_initializer = (checkpoint_ops._load_and_remap_matrix_initializer(
        new_row_vocab_size=5,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.checkpoint_file],
        new_row_vocab_file=self.new_feature_vocab_file,
        old_row_vocab_file=self.old_feature_vocab_file,
        num_row_oov_buckets=5,
        num_col_oov_buckets=1,
        initializer=self.initializer))

    # The new weight matrix is of size
    # [5 feature vocab + 5 feature OOV, 4 class vocab + 1 class OOV].  The
    # second partition has only OOV.
    expected_remapped_matrix = np.concatenate(
        [
            np.reshape([2, 18, 34, 50] + [self.init_val] * 6, [10, 1]),
            np.reshape([0, 16, 32, 48] + [self.init_val] * 6, [10, 1]),
            np.reshape([self.init_val] * 10, [10, 1]),
            np.reshape([1, 17, 33, 49] + [self.init_val] * 6, [10, 1]),
            np.reshape([self.init_val] * 10, [10, 1]),
        ],
        axis=1)
    remapped_matrix = variable_scope.get_variable(
        name='linear_all_oov/obtained_weight_matrix',
        shape=[10, 5],
        initializer=loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(expected_remapped_matrix, remapped_matrix.as_tensor())

  def test_load_and_remap_linear_multiclass_initializer_default_init(self):
    # [FUNCTION DOC]
    # Scenario: same setup as the linear multi-class case but no initializer is passed, so the
    # zeros default applies. Action: build the initializer and create a 6 by 5 variable in two
    # row partitions. Expected: remapped values are loaded from the checkpoint and every missing
    # or OOV entry is zero.
    """Tests where the zeros_initializer default is used for linear."""
    loading_initializer = (checkpoint_ops._load_and_remap_matrix_initializer(
        new_row_vocab_size=5,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.checkpoint_file],
        new_row_vocab_file=self.new_feature_vocab_file,
        old_row_vocab_file=self.old_feature_vocab_file,
        num_row_oov_buckets=1,
        num_col_oov_buckets=1))

    # Same as test_initializer_with_oov_only_partition, but with zero
    # initialization.
    expected_remapped_matrix = np.concatenate(
        [
            np.reshape([2, 18, 34, 50, 0, 0], [6, 1]),
            np.reshape([0, 16, 32, 48, 0, 0], [6, 1]),
            np.reshape([0] * 6, [6, 1]),
            np.reshape([1, 17, 33, 49, 0, 0], [6, 1]),
            np.reshape([0] * 6, [6, 1])
        ],
        axis=1)
    remapped_matrix = variable_scope.get_variable(
        name='linear_init_fallback/obtained_weight_matrix',
        shape=[6, 5],
        initializer=loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(expected_remapped_matrix, remapped_matrix.as_tensor())

  def test_load_embedding_initializer(self):
    # [FUNCTION DOC]
    # Scenario: embedding wrapper with a new vocab of five words (one absent from the old vocab)
    # and one OOV bucket, embedding dimension 16. Action: build the embedding initializer and
    # create a 6 by 16 variable in two partitions. Expected: the first four rows equal the
    # checkpoint rows in order, and the last two rows (the new word and the OOV row) equal the
    # constant fill value.
    """Tests for the load_embedding_initializer wrapper."""
    embedding_loading_initializer = (checkpoint_ops._load_embedding_initializer(
        new_vocab_file=self.new_feature_vocab_file,
        old_vocab_file=self.old_feature_vocab_file,
        new_vocab_size=5,
        embedding_dim=16,
        embedding_tensor_name='some_scope/embeddings',
        ckpt_path=[self.checkpoint_file],
        num_oov_buckets=1,
        initializer=self.initializer))

    # The new weight matrix is of size
    # [5 feature vocab + 1 feature OOV, 16 (embedding dimension)], where the
    # last vocab row (2nd last row) is newly initialized (wasn't found in
    # previous vocab) and the actual last row is OOV and also newly initialized.
    # Use a partitioned variable to confirm that the offset logic works.
    expected_remapped_embeddings = np.concatenate(
        [
            np.reshape(range(64), [4, 16]),
            np.reshape([self.init_val] * 32, [2, 16]),
        ],
        axis=0)
    remapped_embeddings = variable_scope.get_variable(
        name='embedding/obtained_embedding_matrix',
        shape=[6, 16],
        initializer=embedding_loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(expected_remapped_embeddings,
                          remapped_embeddings.as_tensor())

  def test_load_embedding_initializer_large_oov(self):
    # [FUNCTION DOC]
    # Scenario: rewrites the vocab files (new: one, zero, two, four; old: zero, one) and uses
    # five OOV buckets so that most rows are newly initialized. Action: build the embedding
    # initializer and create a 9 by 16 variable in two partitions. Expected: row one is the
    # checkpoint's second row, row two is the first row, and the remaining seven rows (two
    # unknown words and five OOV buckets) hold the constant fill value.
    # Side effect: overwrites the vocabulary files created in setUp.
    """Tests for the large OOV case for load_embedding_initializer wrapper."""
    self.new_feature_vocab_file = os.path.join(
        self.get_temp_dir(), 'new_feature_vocab.txt')
    with open(self.new_feature_vocab_file, 'w') as f:
      f.write('\n'.join(['one', 'zero', 'two', 'four']) + '\n')

    # Checkpoint has 5 entries, 3 of which correspond to OOV.
    self.old_feature_vocab_file = os.path.join(
        self.get_temp_dir(), 'old_feature_vocab.txt')
    with open(self.old_feature_vocab_file, 'w') as f:
      f.write('\n'.join(['zero', 'one']) + '\n')

    embedding_loading_initializer = (checkpoint_ops._load_embedding_initializer(
        new_vocab_file=self.new_feature_vocab_file,
        old_vocab_file=self.old_feature_vocab_file,
        new_vocab_size=4,
        embedding_dim=16,
        embedding_tensor_name='some_scope/embeddings',
        ckpt_path=[self.checkpoint_file],
        num_oov_buckets=5,
        initializer=self.initializer))

    # The new weight matrix is of size
    # [4 feature vocab + 5 feature OOV, 16 (embedding dimension)], where the
    # 3rd and 4th rows are not found in the old vocabulary and therefore newly
    # initialized.  The last five rows are OOV and also newly initialized.
    # Use a partitioned variable to confirm that the offset logic works.
    expected_remapped_embeddings = np.concatenate(
        [
            np.reshape(range(16, 32), [1, 16]),
            np.reshape(range(16), [1, 16]),
            np.reshape([self.init_val] * 112, [7, 16]),
        ],
        axis=0)
    remapped_embeddings = variable_scope.get_variable(
        name='embedding/obtained_embedding_matrix',
        shape=[9, 16],
        initializer=embedding_loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(expected_remapped_embeddings,
                          remapped_embeddings.as_tensor())

  def test_load_embedding_initializer_old_row_vocab(self):
    # [FUNCTION DOC]
    # Scenario: the old vocabulary size is limited to three entries, so the fourth old word is
    # treated as unknown in the new vocab. Action: build the embedding initializer with five new
    # words and one OOV bucket, and create a 6 by 16 variable in two partitions. Expected: the
    # first three rows come from the checkpoint and the last three rows hold the constant fill.
    """Tests for load_embedding_initializer where we constrain old vocab."""
    embedding_loading_initializer = (
        checkpoint_ops._load_embedding_initializer(
            new_vocab_file=self.new_feature_vocab_file,
            old_vocab_file=self.old_feature_vocab_file,
            # Considered old vocabulary becomes ['zero', 'one', 'two'].  This
            # means 'three' in the new vocabulary is newly initialized.
            old_vocab_size=3,
            new_vocab_size=5,
            embedding_dim=16,
            embedding_tensor_name='some_scope/embeddings',
            ckpt_path=[self.checkpoint_file],
            num_oov_buckets=1,
            initializer=self.initializer))

    # The new weight matrix is of size
    # [5 feature vocab + 1 feature OOV, 16 (embedding dimension)], where the
    # last vocab row (2nd last row) is newly initialized (wasn't found in
    # previous vocab) and the actual last row is OOV and also newly initialized.
    # Use a partitioned variable to confirm that the offset logic works.
    expected_remapped_embeddings = np.concatenate(
        [
            np.reshape(range(48), [3, 16]),
            np.reshape([self.init_val] * 48, [3, 16]),
        ],
        axis=0)
    remapped_embeddings = variable_scope.get_variable(
        name='embedding/obtained_embedding_matrix',
        shape=[6, 16],
        initializer=embedding_loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(expected_remapped_embeddings,
                          remapped_embeddings.as_tensor())


if __name__ == '__main__':
  test.main()
