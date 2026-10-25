# [MODULE DOC]
# Unit tests for warm_starting_util: the VocabInfo record, the private variable-info and
# vocabulary remapping helpers, and the public warm_start function. A single test class writes
# small checkpoints (plain, partitioned, with vocabulary files, object-based) in one graph and
# then warm-starts variables of a fresh graph, including linear-model and embedding feature
# column variables, checking the resulting variable values. Uses the temporary directory of
# the test case for checkpoints and vocabulary files. No import-time side effects besides
# defining shorthand initializers.
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
"""Tests for warm_starting_util."""

import os

import numpy as np

from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import warm_starting_util as ws_util

ones = init_ops.ones_initializer
norms = init_ops.truncated_normal_initializer
rand = init_ops.random_uniform_initializer
zeros = init_ops.zeros_initializer


class WarmStartingUtilTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering warm-starting from checkpoints. Provides helper methods to write
  # vocabulary files and checkpoints (with optionally partitioned variables) into the test
  # temporary directory, and to build linear models whose per-column variables are inspected.
  # Each test builds a previous-run graph that saves a checkpoint, then a new graph and session
  # in which variables are warm-started and verified.

  def _write_vocab(self, string_values, file_name):
    # [FUNCTION DOC]
    # HELPER: shared fixture for vocabulary tests.
    # Writes the given list of strings, one per line, to a file of the given name inside the
    # test temporary directory. Returns the full path of the file.
    vocab_file = os.path.join(self.get_temp_dir(), file_name)
    with open(vocab_file, "w") as f:
      f.write("\n".join(string_values))
    return vocab_file

  def _write_checkpoint(self, sess):
    # [FUNCTION DOC]
    # HELPER: shared fixture that saves a checkpoint.
    # Input: sess, the active session. Initializes all global variables, then saves them with a
    # standard saver under the file prefix "model" in the test temporary directory with global
    # step zero (so the file is "model-0"). Returns nothing.
    self.evaluate(variables.global_variables_initializer())
    saver = saver_lib.Saver()
    ckpt_prefix = os.path.join(self.get_temp_dir(), "model")
    saver.save(sess, ckpt_prefix, global_step=0)

  def _create_prev_run_var(self,
                           var_name,
                           shape=None,
                           initializer=None,
                           partitioner=None):
    # [FUNCTION DOC]
    # HELPER: simulates a previous run that saved one variable.
    # Inputs: var_name, optional shape, initializer and partitioner. In a private graph creates
    # the variable, writes a checkpoint, and if partitioned asserts the variable is a
    # partitioned variable and replaces it by its list of slices. Returns a pair: the variable
    # (or list of slices) and its evaluated value(s).
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        var = variable_scope.get_variable(
            var_name,
            shape=shape,
            initializer=initializer,
            partitioner=partitioner)
        self._write_checkpoint(sess)
        if partitioner:
          self.assertTrue(isinstance(var, variables.PartitionedVariable))
          var = var._get_variable_list()
        return var, self.evaluate(var)

  def _create_prev_run_vars(self,
                            var_names,
                            shapes,
                            initializers):
    # [FUNCTION DOC]
    # HELPER: like the single-variable fixture but for several variables.
    # Inputs: parallel lists of variable names, shapes and initializers. In a private graph
    # creates all variables, writes a checkpoint, and returns the list of evaluated values in
    # the same order.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        all_vars = []
        for var_name, shape, initializer in zip(var_names, shapes,
                                                initializers):
          all_vars.append(variable_scope.get_variable(
              var_name,
              shape=shape,
              initializer=initializer))
        self._write_checkpoint(sess)
        return [self.evaluate(var) for var in all_vars]

  def _create_dummy_inputs(self):
    # [FUNCTION DOC]
    # HELPER: builds placeholder inputs for feature columns. Returns a mapping with sparse
    # placeholders named for integer, hashed, key-list and vocabulary columns and a dense
    # float placeholder named "real".
    return {
        "sc_int": array_ops.sparse_placeholder(dtypes.int32),
        "sc_hash": array_ops.sparse_placeholder(dtypes.string),
        "sc_keys": array_ops.sparse_placeholder(dtypes.string),
        "sc_vocab": array_ops.sparse_placeholder(dtypes.string),
        "real": array_ops.placeholder(dtypes.float32)
    }

  def _create_linear_model(self, feature_cols, partitioner):
    # [FUNCTION DOC]
    # HELPER: creates the variables of a linear model over the given feature columns.
    # Inputs: feature_cols (list of columns) and partitioner (function or nothing) applied in
    # the root variable scope. Builds the model on dummy inputs with one unit. Returns the
    # mapping from each column (and the bias) to the list of its variables.
    cols_to_vars = {}
    with variable_scope.variable_scope("", partitioner=partitioner):
      # Create the variables.
      fc.linear_model(
          features=self._create_dummy_inputs(),
          feature_columns=feature_cols,
          units=1,
          cols_to_vars=cols_to_vars)
    # Return a dictionary mapping each column to its variable.
    return cols_to_vars

  def _assert_cols_to_vars(self, cols_to_vars, cols_to_expected_values, sess):
    # [FUNCTION DOC]
    # HELPER: assertion shared by the feature column tests.
    # Inputs: cols_to_vars (column to list of variables), cols_to_expected_values (column to
    # list of expected arrays, one per variable slice) and sess (session). Evaluates each
    # variable and asserts it is close to the expected array. Returns nothing.
    for col, expected_values in cols_to_expected_values.items():
      for i, var in enumerate(cols_to_vars[col]):
        self.assertAllClose(expected_values[i], var.eval(sess))

  def testWarmStartVar(self):
    # [FUNCTION DOC]
    # Scenario: a checkpoint holds a 4x1 variable. In a new graph a same-named zero variable is
    # created. Action: the variable info helper and checkpoint initialization are applied.
    # Expected: after initialization the variable equals the checkpointed values.
    _, prev_val = self._create_prev_run_var(
        "fruit_weights", initializer=[[0.5], [1.], [1.5], [2.]])

    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights", initializer=[[0.], [0.], [0.], [0.]])
        prev_tensor_name, var = ws_util._get_var_info(fruit_weights)
        checkpoint_utils.init_from_checkpoint(self.get_temp_dir(),
                                              {prev_tensor_name: var})
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose(prev_val, fruit_weights.eval(sess))

  def testWarmStartVarPrevVarPartitioned(self):
    # [FUNCTION DOC]
    # Scenario: the checkpoint variable was saved in two partitions; the new variable is a
    # single zero variable. Expected: after variable-info plus checkpoint initialization the new
    # variable equals the concatenation of the saved partitions.
    _, weights = self._create_prev_run_var(
        "fruit_weights",
        shape=[4, 1],
        initializer=[[0.5], [1.], [1.5], [2.]],
        partitioner=lambda shape, dtype: [2, 1])
    prev_val = np.concatenate([weights[0], weights[1]], axis=0)

    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights", initializer=[[0.], [0.], [0.], [0.]])
        prev_tensor_name, var = ws_util._get_var_info(fruit_weights)
        checkpoint_utils.init_from_checkpoint(self.get_temp_dir(),
                                              {prev_tensor_name: var})
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose(prev_val, fruit_weights.eval(sess))

  def testWarmStartVarCurrentVarPartitioned(self):
    # [FUNCTION DOC]
    # Scenario: the checkpoint holds an unpartitioned 4x1 variable; the new variable is
    # partitioned into two slices. Expected: after initialization the concatenated slices equal
    # the saved values.
    _, prev_val = self._create_prev_run_var(
        "fruit_weights", initializer=[[0.5], [1.], [1.5], [2.]])

    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights",
            shape=[4, 1],
            initializer=[[0.], [0.], [0.], [0.]],
            partitioner=lambda shape, dtype: [2, 1])
        self.assertTrue(
            isinstance(fruit_weights, variables.PartitionedVariable))
        prev_tensor_name, var = ws_util._get_var_info(fruit_weights)
        checkpoint_utils.init_from_checkpoint(self.get_temp_dir(),
                                              {prev_tensor_name: var})
        self.evaluate(variables.global_variables_initializer())
        fruit_weights = fruit_weights._get_variable_list()
        new_val = np.concatenate(
            [fruit_weights[0].eval(sess), fruit_weights[1].eval(sess)], axis=0)
        self.assertAllClose(prev_val, new_val)

  def testWarmStartVarBothVarsPartitioned(self):
    # [FUNCTION DOC]
    # Scenario: both the saved and the new variable are partitioned in two, under different
    # scope names. The previous tensor name is passed explicitly to the variable info helper.
    # Expected: the new variable's concatenated slices equal the saved values.
    _, weights = self._create_prev_run_var(
        "old_scope/fruit_weights",
        shape=[4, 1],
        initializer=[[0.5], [1.], [1.5], [2.]],
        partitioner=lambda shape, dtype: [2, 1])
    prev_val = np.concatenate([weights[0], weights[1]], axis=0)
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "new_scope/fruit_weights",
            shape=[4, 1],
            initializer=[[0.], [0.], [0.], [0.]],
            partitioner=lambda shape, dtype: [2, 1])
        self.assertTrue(
            isinstance(fruit_weights, variables.PartitionedVariable))
        prev_tensor_name, var = ws_util._get_var_info(
            fruit_weights, prev_tensor_name="old_scope/fruit_weights")
        checkpoint_utils.init_from_checkpoint(self.get_temp_dir(),
                                              {prev_tensor_name: var})
        self.evaluate(variables.global_variables_initializer())
        fruit_weights = fruit_weights._get_variable_list()
        new_val = np.concatenate(
            [fruit_weights[0].eval(sess), fruit_weights[1].eval(sess)], axis=0)
        self.assertAllClose(prev_val, new_val)

  def testWarmStartVarWithVocab(self):
    # [FUNCTION DOC]
    # Scenario: old vocabulary of four fruits with a saved 4x1 weight; new vocabulary lists them
    # in reverse order plus one new fruit (5 entries). Action: the vocabulary warm-start helper
    # is applied to a zero 5x1 variable. Expected: rows are reordered by word to
    # 2, 1.5, 1, 0.5 and the new word gets zero.
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    self._create_prev_run_var(
        "fruit_weights", initializer=[[0.5], [1.], [1.5], [2.]])

    # New vocab with elements in reverse order and one new element.
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry"], "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights", initializer=[[0.], [0.], [0.], [0.], [0.]])
        ws_util._warm_start_var_with_vocab(fruit_weights, new_vocab_path, 5,
                                           self.get_temp_dir(), prev_vocab_path)
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose([[2.], [1.5], [1.], [0.5], [0.]],
                            fruit_weights.eval(sess))

  def testWarmStartVarWithColumnVocab(self):
    # [FUNCTION DOC]
    # Scenario: a saved 4x2 output layer whose columns correspond to two old vocabulary words;
    # the new vocabulary has three words (swapped order plus a new one). Action: the vocabulary
    # helper with axis one. Expected: columns are swapped by word and the new column is zero.
    prev_vocab_path = self._write_vocab(["apple", "orange"], "old_vocab")
    self._create_prev_run_var(
        "fruit_output_layer",
        initializer=[[0.5, 0.3], [1., 0.8], [1.5, 1.2], [2., 2.3]])

    # New vocab with elements in reverse order and one new element.
    new_vocab_path = self._write_vocab(["orange", "apple", "banana"],
                                       "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        fruit_output_layer = variable_scope.get_variable(
            "fruit_output_layer",
            initializer=[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                         [0., 0., 0.]])
        ws_util._warm_start_var_with_vocab(fruit_output_layer, new_vocab_path,
                                           current_vocab_size=3,
                                           prev_ckpt=self.get_temp_dir(),
                                           prev_vocab_path=prev_vocab_path,
                                           axis=1)
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose([[0.3, 0.5, 0.], [0.8, 1.0, 0.], [1.2, 1.5, 0.],
                             [2.3, 2., 0.]], fruit_output_layer.eval(sess))

  def testWarmStartVarWithVocabConstrainedOldVocabSize(self):
    # [FUNCTION DOC]
    # Scenario: same as the basic vocabulary test but the previous vocabulary size is limited to
    # the first two words. Expected: only words in those two entries are carried over
    # (banana gets 1, apple gets 0.5); others, including guava and orange, stay zero.
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    self._create_prev_run_var(
        "fruit_weights", initializer=[[0.5], [1.], [1.5], [2.]])

    # New vocab with elements in reverse order and one new element.
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry"], "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights", initializer=[[0.], [0.], [0.], [0.], [0.]])
        ws_util._warm_start_var_with_vocab(
            fruit_weights,
            new_vocab_path,
            5,
            self.get_temp_dir(),
            prev_vocab_path,
            previous_vocab_size=2)
        self.evaluate(variables.global_variables_initializer())
        # Old vocabulary limited to ['apple', 'banana'].
        self.assertAllClose([[0.], [0.], [1.], [0.5], [0.]],
                            fruit_weights.eval(sess))

  def testWarmStartVarWithVocabPrevVarPartitioned(self):
    # [FUNCTION DOC]
    # Scenario: the vocabulary-test setup, but the saved variable was written in two partitions.
    # Expected: the remapped result is unchanged: reversed order values with zero for the new
    # word.
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    self._create_prev_run_var(
        "fruit_weights",
        shape=[4, 1],
        initializer=[[0.5], [1.], [1.5], [2.]],
        partitioner=lambda shape, dtype: [2, 1])

    # New vocab with elements in reverse order and one new element.
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry"], "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights", initializer=[[0.], [0.], [0.], [0.], [0.]])
        ws_util._warm_start_var_with_vocab(fruit_weights, new_vocab_path, 5,
                                           self.get_temp_dir(), prev_vocab_path)
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose([[2.], [1.5], [1.], [0.5], [0.]],
                            fruit_weights.eval(sess))

  def testWarmStartVarWithColumnVocabPrevVarPartitioned(self):
    # [FUNCTION DOC]
    # Scenario: the column-vocabulary setup (axis one) with the saved 4x2 variable partitioned in
    # two. Expected: columns remapped by word as in the unpartitioned test, new column zero.
    prev_vocab_path = self._write_vocab(["apple", "orange"], "old_vocab")
    self._create_prev_run_var(
        "fruit_output_layer",
        shape=[4, 2],
        initializer=[[0.5, 0.3], [1., 0.8], [1.5, 1.2], [2., 2.3]],
        partitioner=lambda shape, dtype: [2, 1])

    # New vocab with elements in reverse order and one new element.
    new_vocab_path = self._write_vocab(["orange", "apple", "banana"],
                                       "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        fruit_output_layer = variable_scope.get_variable(
            "fruit_output_layer",
            initializer=[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                         [0., 0., 0.]])
        ws_util._warm_start_var_with_vocab(fruit_output_layer, new_vocab_path,
                                           current_vocab_size=3,
                                           prev_ckpt=self.get_temp_dir(),
                                           prev_vocab_path=prev_vocab_path,
                                           axis=1)
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose([[0.3, 0.5, 0.], [0.8, 1.0, 0.], [1.2, 1.5, 0.],
                             [2.3, 2., 0.]], fruit_output_layer.eval(sess))

  def testWarmStartVarWithVocabCurrentVarPartitioned(self):
    # [FUNCTION DOC]
    # Scenario: the new variable has six rows in two partitions (five vocabulary entries plus
    # one out-of-vocabulary bucket). Expected: the first slice holds 2, 1.5, 1 and the second
    # 0.5 followed by two zeros; the variable remains a partitioned variable.
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    self._create_prev_run_var(
        "fruit_weights", initializer=[[0.5], [1.], [1.5], [2.]])

    # New vocab with elements in reverse order and one new element.
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry"], "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights",
            shape=[6, 1],
            initializer=[[0.], [0.], [0.], [0.], [0.], [0.]],
            partitioner=lambda shape, dtype: [2, 1])
        ws_util._warm_start_var_with_vocab(
            fruit_weights,
            new_vocab_path,
            5,
            self.get_temp_dir(),
            prev_vocab_path,
            current_oov_buckets=1)
        self.evaluate(variables.global_variables_initializer())
        self.assertTrue(
            isinstance(fruit_weights, variables.PartitionedVariable))
        fruit_weights_vars = fruit_weights._get_variable_list()
        self.assertAllClose([[2.], [1.5], [1.]],
                            fruit_weights_vars[0].eval(sess))
        self.assertAllClose([[0.5], [0.], [0.]],
                            fruit_weights_vars[1].eval(sess))

  def testWarmStartVarWithColumnVocabCurrentVarPartitioned(self):
    # [FUNCTION DOC]
    # Scenario: column-vocabulary remapping into a 4x3 variable partitioned into two row
    # slices. Expected: each slice has the swapped columns with a zero last column
    # (first slice rows 0.3/0.5 and 0.8/1.0, second slice rows 1.2/1.5 and 2.3/2.0).
    prev_vocab_path = self._write_vocab(["apple", "orange"], "old_vocab")
    self._create_prev_run_var(
        "fruit_output_layer",
        initializer=[[0.5, 0.3], [1., 0.8], [1.5, 1.2], [2., 2.3]])

    # New vocab with elements in reverse order and one new element.
    new_vocab_path = self._write_vocab(["orange", "apple", "banana"],
                                       "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        fruit_output_layer = variable_scope.get_variable(
            "fruit_output_layer",
            shape=[4, 3],
            initializer=[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                         [0., 0., 0.]],
            partitioner=lambda shape, dtype: [2, 1])
        ws_util._warm_start_var_with_vocab(fruit_output_layer, new_vocab_path,
                                           current_vocab_size=3,
                                           prev_ckpt=self.get_temp_dir(),
                                           prev_vocab_path=prev_vocab_path,
                                           axis=1)
        self.evaluate(variables.global_variables_initializer())
        self.assertTrue(
            isinstance(fruit_output_layer, variables.PartitionedVariable))
        fruit_output_layer_vars = fruit_output_layer._get_variable_list()
        self.assertAllClose([[0.3, 0.5, 0.], [0.8, 1.0, 0.]],
                            fruit_output_layer_vars[0].eval(sess))
        self.assertAllClose([[1.2, 1.5, 0.], [2.3, 2., 0.]],
                            fruit_output_layer_vars[1].eval(sess))

  def testWarmStartVarWithVocabBothVarsPartitioned(self):
    # [FUNCTION DOC]
    # Scenario: saved variable partitioned in two; new vocabulary has six words and the new
    # variable is six rows in two partitions. Expected: first slice 2, 1.5, 1; second slice 0.5,
    # 0, 0 (both new words are zero).
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    self._create_prev_run_var(
        "fruit_weights",
        shape=[4, 1],
        initializer=[[0.5], [1.], [1.5], [2.]],
        partitioner=lambda shape, dtype: [2, 1])

    # New vocab with elements in reverse order and two new elements.
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry",
         "blueberry"], "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights",
            shape=[6, 1],
            initializer=[[0.], [0.], [0.], [0.], [0.], [0.]],
            partitioner=lambda shape, dtype: [2, 1])
        ws_util._warm_start_var_with_vocab(fruit_weights, new_vocab_path, 6,
                                           self.get_temp_dir(), prev_vocab_path)
        self.evaluate(variables.global_variables_initializer())
        self.assertTrue(
            isinstance(fruit_weights, variables.PartitionedVariable))
        fruit_weights_vars = fruit_weights._get_variable_list()
        self.assertAllClose([[2.], [1.5], [1.]],
                            fruit_weights_vars[0].eval(sess))
        self.assertAllClose([[0.5], [0.], [0.]],
                            fruit_weights_vars[1].eval(sess))

  def testWarmStartVarWithColumnVocabBothVarsPartitioned(self):
    # [FUNCTION DOC]
    # Scenario: column-vocabulary remapping where both the saved 4x2 variable and the new 4x3
    # variable are split into two row partitions. Expected: swapped columns with zero third
    # column in each slice.
    prev_vocab_path = self._write_vocab(["apple", "orange"], "old_vocab")
    self._create_prev_run_var(
        "fruit_output_layer",
        shape=[4, 2],
        initializer=[[0.5, 0.3], [1., 0.8], [1.5, 1.2], [2., 2.3]],
        partitioner=lambda shape, dtype: [2, 1])

    # New vocab with elements in reverse order and one new element.
    new_vocab_path = self._write_vocab(["orange", "apple", "banana"],
                                       "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        fruit_output_layer = variable_scope.get_variable(
            "fruit_output_layer",
            shape=[4, 3],
            initializer=[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                         [0., 0., 0.]],
            partitioner=lambda shape, dtype: [2, 1])
        ws_util._warm_start_var_with_vocab(fruit_output_layer, new_vocab_path,
                                           current_vocab_size=3,
                                           prev_ckpt=self.get_temp_dir(),
                                           prev_vocab_path=prev_vocab_path,
                                           axis=1)
        self.evaluate(variables.global_variables_initializer())
        self.assertTrue(
            isinstance(fruit_output_layer, variables.PartitionedVariable))
        fruit_output_layer_vars = fruit_output_layer._get_variable_list()
        self.assertAllClose([[0.3, 0.5, 0.], [0.8, 1.0, 0.]],
                            fruit_output_layer_vars[0].eval(sess))
        self.assertAllClose([[1.2, 1.5, 0.], [2.3, 2., 0.]],
                            fruit_output_layer_vars[1].eval(sess))

  def testWarmStart_ListOfVariables(self):
    # [FUNCTION DOC]
    # Scenario: checkpoint saves a 10x1 variable of ones; new graph has a zero variable of the
    # same name. Action: warm start with the list of variable objects. Expected: after
    # initialization the variable equals ones.
    # Save checkpoint from which to warm-start.
    _, prev_int_val = self._create_prev_run_var("v1", shape=[10, 1],
                                                initializer=ones())
    # Verify we initialized the values correctly.
    self.assertAllEqual(np.ones([10, 1]), prev_int_val)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        # Initialize with zeros.
        var = variable_scope.get_variable(
            "v1",
            shape=[10, 1],
            initializer=zeros())
        ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start=[var])
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started (init overridden to ones).
        self.assertAllEqual(var, prev_int_val)

  def testWarmStart_ListOfStrings(self):
    # [FUNCTION DOC]
    # Scenario: same as the list-of-variables test but the variables to warm start are given as
    # a list containing the variable name string. Expected: the variable is initialized to ones.
    # Save checkpoint from which to warm-start.
    _, prev_int_val = self._create_prev_run_var("v1", shape=[10, 1],
                                                initializer=ones())
    # Verify we initialized the values correctly.
    self.assertAllEqual(np.ones([10, 1]), prev_int_val)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        # Initialize with zeros.
        var = variable_scope.get_variable(
            "v1",
            shape=[10, 1],
            initializer=zeros())
        ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start=["v1"])
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started (init overridden to ones).
        self.assertAllEqual(var, prev_int_val)

  def testWarmStart_TwoVarsFromTheSamePrevVar(self):
    # [FUNCTION DOC]
    # Scenario: the checkpoint has only "v1"; the new graph has zero variables "v1" and "v2" and
    # a rename mapping from "v2" to "v1". Action: warm start both. Expected: both variables are
    # initialized to ones, exercising the split initialization for a shared source tensor.
    # Save checkpoint from which to warm-start.
    _, prev_int_val = self._create_prev_run_var("v1", shape=[10, 1],
                                                initializer=ones())
    # Verify we initialized the values correctly.
    self.assertAllEqual(np.ones([10, 1]), prev_int_val)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        # Initialize with zeros.
        var = variable_scope.get_variable(
            "v1",
            shape=[10, 1],
            initializer=zeros())
        var2 = variable_scope.get_variable(
            "v2",
            shape=[10, 1],
            initializer=zeros())
        ws_util.warm_start(self.get_temp_dir(),
                           vars_to_warm_start=["v1", "v2"],
                           var_name_to_prev_var_name=dict(v2="v1"))
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started (init overridden to ones).
        self.assertAllEqual(var, prev_int_val)
        self.assertAllEqual(var2, prev_int_val)

  def testWarmStart_ListOfRegexes(self):
    # [FUNCTION DOC]
    # Scenario: checkpoint holds v1, v1/Momentum, v2 and v2/Momentum (ones); the new graph has
    # zero versions of all four. Action: warm start with scope patterns for v1 and for v2
    # followed by a non-slash character. Expected: v1, v1/Momentum and v2 become ones while
    # v2/Momentum stays zero.
    # Save checkpoint from which to warm-start.
    [prev_v1_val, prev_v1_momentum_val,
     prev_v2_val, _] = self._create_prev_run_vars(
         var_names=["v1", "v1/Momentum", "v2", "v2/Momentum"],
         shapes=[[10, 1]] * 4,
         initializers=[ones()] * 4)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        # Initialize with zeros.
        v1 = variable_scope.get_variable(
            "v1",
            shape=[10, 1],
            initializer=zeros())
        v1_momentum = variable_scope.get_variable(
            "v1/Momentum",
            shape=[10, 1],
            initializer=zeros())
        v2 = variable_scope.get_variable(
            "v2",
            shape=[10, 1],
            initializer=zeros())
        v2_momentum = variable_scope.get_variable(
            "v2/Momentum",
            shape=[10, 1],
            initializer=zeros())
        ws_util.warm_start(self.get_temp_dir(),
                           # This warm-starts both v1 and v1/Momentum, but only
                           # v2 (and not v2/Momentum).
                           vars_to_warm_start=["v1", "v2[^/]"])
        self.evaluate(variables.global_variables_initializer())
        # Verify the selection of weights were correctly warm-started (init
        # overridden to ones).
        self.assertAllEqual(v1, prev_v1_val)
        self.assertAllEqual(v1_momentum, prev_v1_momentum_val)
        self.assertAllEqual(v2, prev_v2_val)
        self.assertAllEqual(v2_momentum, np.zeros([10, 1]))

  def testWarmStart_SparseColumnIntegerized(self):
    # [FUNCTION DOC]
    # Scenario: identity categorical column with ten buckets in a linear model; a checkpoint
    # holds its weights as ones. First verifies a model without warm start is zero, then warm
    # starts with a regular expression for the column. Expected: weights equal the saved ones.
    # Create feature column.
    sc_int = fc.categorical_column_with_identity("sc_int", num_buckets=10)

    # Save checkpoint from which to warm-start.
    _, prev_int_val = self._create_prev_run_var(
        "linear_model/sc_int/weights", shape=[10, 1], initializer=ones())
    # Verify we initialized the values correctly.
    self.assertAllEqual(np.ones([10, 1]), prev_int_val)

    partitioner = lambda shape, dtype: [1] * len(shape)
    # New graph, new session WITHOUT warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_int], partitioner)
        self.evaluate(variables.global_variables_initializer())
        # Without warm-starting, the weights should be initialized using default
        # initializer (which is init_ops.zeros_initializer).
        self._assert_cols_to_vars(cols_to_vars, {sc_int: [np.zeros([10, 1])]},
                                  sess)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_int], partitioner)
        ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start=".*sc_int.*")
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started.
        self._assert_cols_to_vars(cols_to_vars, {sc_int: [prev_int_val]}, sess)

  def testWarmStart_SparseColumnHashed(self):
    # [FUNCTION DOC]
    # Scenario: hashed categorical column of 15 buckets with saved random weights. Without warm
    # start the weights are zeros; with warm start by a pattern for the column they equal the
    # saved values.
    # Create feature column.
    sc_hash = fc.categorical_column_with_hash_bucket(
        "sc_hash", hash_bucket_size=15)

    # Save checkpoint from which to warm-start.
    _, prev_hash_val = self._create_prev_run_var(
        "linear_model/sc_hash/weights", shape=[15, 1], initializer=norms())

    partitioner = lambda shape, dtype: [1] * len(shape)
    # New graph, new session WITHOUT warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_hash], partitioner)
        self.evaluate(variables.global_variables_initializer())
        # Without warm-starting, the weights should be initialized using default
        # initializer (which is init_ops.zeros_initializer).
        self._assert_cols_to_vars(cols_to_vars, {sc_hash: [np.zeros([15, 1])]},
                                  sess)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_hash], partitioner)
        ws_util.warm_start(
            self.get_temp_dir(), vars_to_warm_start=".*sc_hash.*")
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started.
        self._assert_cols_to_vars(cols_to_vars, {sc_hash: [prev_hash_val]},
                                  sess)

  def testWarmStart_SparseColumnVocabulary(self):
    # [FUNCTION DOC]
    # Scenario: vocabulary-file categorical column (four words) with saved weights of ones.
    # Without warm start weights are zero; with warm start by pattern and no vocabulary info
    # (old vocabulary assumed unchanged) the weights equal the saved values.
    # Create vocab for sparse column "sc_vocab".
    vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                   "vocab")
    # Create feature column.
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=vocab_path, vocabulary_size=4)

    # Save checkpoint from which to warm-start.
    _, prev_vocab_val = self._create_prev_run_var(
        "linear_model/sc_vocab/weights", shape=[4, 1], initializer=ones())

    partitioner = lambda shape, dtype: [1] * len(shape)
    # New graph, new session WITHOUT warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
        self.evaluate(variables.global_variables_initializer())
        # Without warm-starting, the weights should be initialized using default
        # initializer (which is init_ops.zeros_initializer).
        self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [np.zeros([4, 1])]},
                                  sess)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
        # Since old vocab is not explicitly set in WarmStartSettings, the old
        # vocab is assumed to be same as new vocab.
        ws_util.warm_start(
            self.get_temp_dir(), vars_to_warm_start=".*sc_vocab.*")
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started.
        self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [prev_vocab_val]},
                                  sess)

  def testWarmStart_ExplicitCheckpointFile(self):
    # [FUNCTION DOC]
    # Scenario: same as the vocabulary column test, but warm start is given the explicit
    # checkpoint file prefix ("model-0") instead of the directory. Expected: weights are
    # warm-started to the saved values.
    # Create vocab for sparse column "sc_vocab".
    vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                   "vocab")
    # Create feature column.
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=vocab_path, vocabulary_size=4)

    # Save checkpoint from which to warm-start.
    _, prev_vocab_val = self._create_prev_run_var(
        "linear_model/sc_vocab/weights", shape=[4, 1], initializer=ones())

    partitioner = lambda shape, dtype: [1] * len(shape)
    # New graph, new session WITHOUT warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
        self.evaluate(variables.global_variables_initializer())
        # Without warm-starting, the weights should be initialized using default
        # initializer (which is init_ops.zeros_initializer).
        self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [np.zeros([4, 1])]},
                                  sess)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
        # Since old vocab is not explicitly set in WarmStartSettings, the old
        # vocab is assumed to be same as new vocab.
        ws_util.warm_start(
            # Explicitly provide the file prefix instead of just the dir.
            os.path.join(self.get_temp_dir(), "model-0"),
            vars_to_warm_start=".*sc_vocab.*")
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started.
        self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [prev_vocab_val]},
                                  sess)

  def testWarmStart_SparseColumnVocabularyConstrainedVocabSizes(self):
    # [FUNCTION DOC]
    # Scenario: old vocabulary file has three words but only the first two are used (old size
    # two); the new column uses the first two entries of a different four-word file (apple,
    # banana). Vocabulary info with the limited old size is passed to warm start. Expected:
    # apple keeps its saved weight one, banana is not in the old limited vocabulary and is zero.
    # Create old vocabulary, and use a size smaller than the total number of
    # entries.
    old_vocab_path = self._write_vocab(["apple", "guava", "banana"],
                                       "old_vocab")
    old_vocab_size = 2  # ['apple', 'guava']

    # Create new vocab for sparse column "sc_vocab".
    current_vocab_path = self._write_vocab(
        ["apple", "banana", "guava", "orange"], "current_vocab")
    # Create feature column.  Only use 2 of the actual entries, resulting in
    # ['apple', 'banana'] for the new vocabulary.
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=current_vocab_path, vocabulary_size=2)

    # Save checkpoint from which to warm-start.
    self._create_prev_run_var(
        "linear_model/sc_vocab/weights", shape=[2, 1], initializer=ones())

    partitioner = lambda shape, dtype: [1] * len(shape)
    # New graph, new session WITHOUT warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
        self.evaluate(variables.global_variables_initializer())
        # Without warm-starting, the weights should be initialized using default
        # initializer (which is init_ops.zeros_initializer).
        self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [np.zeros([2, 1])]},
                                  sess)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
        vocab_info = ws_util.VocabInfo(
            new_vocab=sc_vocab.vocabulary_file,
            new_vocab_size=sc_vocab.vocabulary_size,
            num_oov_buckets=sc_vocab.num_oov_buckets,
            old_vocab=old_vocab_path,
            old_vocab_size=old_vocab_size)
        ws_util.warm_start(
            ckpt_to_initialize_from=self.get_temp_dir(),
            vars_to_warm_start=".*sc_vocab.*",
            var_name_to_vocab_info={
                "linear_model/sc_vocab/weights": vocab_info
            })
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started.  'banana' isn't in the
        # first two entries of the old vocabulary, so it's newly initialized.
        self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [[[1], [0]]]}, sess)

  def testWarmStart_BucketizedColumn(self):
    # [FUNCTION DOC]
    # Scenario: bucketized numeric column with five buckets and saved random weights. Without
    # warm start weights are zero; warm start with a pattern for the bucketized column restores
    # the saved weights.
    # Create feature column.
    real = fc.numeric_column("real")
    real_bucket = fc.bucketized_column(real, boundaries=[0., 1., 2., 3.])

    # Save checkpoint from which to warm-start.
    _, prev_bucket_val = self._create_prev_run_var(
        "linear_model/real_bucketized/weights",
        shape=[5, 1],
        initializer=norms())

    partitioner = lambda shape, dtype: [1] * len(shape)
    # New graph, new session WITHOUT warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([real_bucket], partitioner)
        self.evaluate(variables.global_variables_initializer())
        # Without warm-starting, the weights should be initialized using default
        # initializer (which is init_ops.zeros_initializer).
        self._assert_cols_to_vars(cols_to_vars,
                                  {real_bucket: [np.zeros([5, 1])]}, sess)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([real_bucket], partitioner)
        ws_util.warm_start(
            self.get_temp_dir(), vars_to_warm_start=".*real_bucketized.*")
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started.
        self._assert_cols_to_vars(cols_to_vars,
                                  {real_bucket: [prev_bucket_val]}, sess)

  def testWarmStart_MultipleCols(self):
    # [FUNCTION DOC]
    # Scenario: a linear model with identity, hashed, key-list, vocabulary, bucketized and
    # crossed columns plus a bias; a checkpoint stores all their weights. Without warm start all
    # are zeros. Action: warm start with default variable selection and vocabulary info for the
    # vocabulary column only. Expected: every column's weights and the bias equal the saved ones.
    # Create vocab for sparse column "sc_vocab".
    vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                   "vocab")

    # Create feature columns.
    sc_int = fc.categorical_column_with_identity("sc_int", num_buckets=10)
    sc_hash = fc.categorical_column_with_hash_bucket(
        "sc_hash", hash_bucket_size=15)
    sc_keys = fc.categorical_column_with_vocabulary_list(
        "sc_keys", vocabulary_list=["a", "b", "c", "e"])
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=vocab_path, vocabulary_size=4)
    real = fc.numeric_column("real")
    real_bucket = fc.bucketized_column(real, boundaries=[0., 1., 2., 3.])
    cross = fc.crossed_column([sc_keys, sc_vocab], hash_bucket_size=20)
    all_linear_cols = [sc_int, sc_hash, sc_keys, sc_vocab, real_bucket, cross]

    # Save checkpoint from which to warm-start.  Also create a bias variable,
    # so we can check that it's also warm-started.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        sc_int_weights = variable_scope.get_variable(
            "linear_model/sc_int/weights", shape=[10, 1], initializer=ones())
        sc_hash_weights = variable_scope.get_variable(
            "linear_model/sc_hash/weights", shape=[15, 1], initializer=norms())
        sc_keys_weights = variable_scope.get_variable(
            "linear_model/sc_keys/weights", shape=[4, 1], initializer=rand())
        sc_vocab_weights = variable_scope.get_variable(
            "linear_model/sc_vocab/weights", shape=[4, 1], initializer=ones())
        real_bucket_weights = variable_scope.get_variable(
            "linear_model/real_bucketized/weights",
            shape=[5, 1],
            initializer=norms())
        cross_weights = variable_scope.get_variable(
            "linear_model/sc_keys_X_sc_vocab/weights",
            shape=[20, 1],
            initializer=rand())
        bias = variable_scope.get_variable(
            "linear_model/bias_weights",
            shape=[1],
            initializer=rand())
        self._write_checkpoint(sess)
        (prev_int_val, prev_hash_val, prev_keys_val, prev_vocab_val,
         prev_bucket_val, prev_cross_val, prev_bias_val) = sess.run([
             sc_int_weights, sc_hash_weights, sc_keys_weights, sc_vocab_weights,
             real_bucket_weights, cross_weights, bias
         ])

    partitioner = lambda shape, dtype: [1] * len(shape)
    # New graph, new session WITHOUT warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model(all_linear_cols, partitioner)
        self.evaluate(variables.global_variables_initializer())
        # Without warm-starting, all weights should be initialized using default
        # initializer (which is init_ops.zeros_initializer).
        self._assert_cols_to_vars(cols_to_vars, {
            sc_int: [np.zeros([10, 1])],
            sc_hash: [np.zeros([15, 1])],
            sc_keys: [np.zeros([4, 1])],
            sc_vocab: [np.zeros([4, 1])],
            real_bucket: [np.zeros([5, 1])],
            cross: [np.zeros([20, 1])],
        }, sess)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model(all_linear_cols, partitioner)
        vocab_info = ws_util.VocabInfo(
            new_vocab=sc_vocab.vocabulary_file,
            new_vocab_size=sc_vocab.vocabulary_size,
            num_oov_buckets=sc_vocab.num_oov_buckets,
            old_vocab=vocab_path)
        ws_util.warm_start(
            self.get_temp_dir(),
            var_name_to_vocab_info={
                "linear_model/sc_vocab/weights": vocab_info
            })
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started.
        self._assert_cols_to_vars(cols_to_vars, {
            sc_int: [prev_int_val],
            sc_hash: [prev_hash_val],
            sc_keys: [prev_keys_val],
            sc_vocab: [prev_vocab_val],
            real_bucket: [prev_bucket_val],
            cross: [prev_cross_val],
            "bias": [prev_bias_val],
        }, sess)

  def testWarmStartMoreSettings(self):
    # [FUNCTION DOC]
    # Scenario: hashed, key-list and vocabulary columns with all variables partitioned in two. The
    # checkpoint stores the key-list weights under a different name. Action: warm start only the
    # key-list and vocabulary variables, with vocabulary info (reordered six-word vocabulary)
    # and a rename for the key-list variable. Expected: key-list weights loaded from the other
    # name, vocabulary weights remapped (3, 2, 1 | 0.5, 0, 0), hashed weights stay zero.
    # Create old and new vocabs for sparse column "sc_vocab".
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry",
         "blueberry"], "new_vocab")
    # Create feature columns.
    sc_hash = fc.categorical_column_with_hash_bucket(
        "sc_hash", hash_bucket_size=15)
    sc_keys = fc.categorical_column_with_vocabulary_list(
        "sc_keys", vocabulary_list=["a", "b", "c", "e"])
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=new_vocab_path, vocabulary_size=6)
    all_linear_cols = [sc_hash, sc_keys, sc_vocab]

    # Save checkpoint from which to warm-start.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        variable_scope.get_variable(
            "linear_model/sc_hash/weights", shape=[15, 1], initializer=norms())
        sc_keys_weights = variable_scope.get_variable(
            "some_other_name", shape=[4, 1], initializer=rand())
        variable_scope.get_variable(
            "linear_model/sc_vocab/weights",
            initializer=[[0.5], [1.], [2.], [3.]])
        self._write_checkpoint(sess)
        prev_keys_val = self.evaluate(sc_keys_weights)

    def _partitioner(shape, dtype):  # pylint:disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: partitioner for the test's model. Inputs: shape and dtype (dtype unused).
      # Returns a per-dimension partition count list that splits the first dimension in two
      # (or fewer if smaller) and leaves other dimensions whole.
      # Partition each var into 2 equal slices.
      partitions = [1] * len(shape)
      partitions[0] = min(2, shape.dims[0].value)
      return partitions

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model(all_linear_cols, _partitioner)
        vocab_info = ws_util.VocabInfo(
            new_vocab=sc_vocab.vocabulary_file,
            new_vocab_size=sc_vocab.vocabulary_size,
            num_oov_buckets=sc_vocab.num_oov_buckets,
            old_vocab=prev_vocab_path)
        ws_util.warm_start(
            self.get_temp_dir(),
            vars_to_warm_start=".*(sc_keys|sc_vocab).*",
            var_name_to_vocab_info={
                ws_util._infer_var_name(cols_to_vars[sc_vocab]): vocab_info
            },
            var_name_to_prev_var_name={
                ws_util._infer_var_name(cols_to_vars[sc_keys]):
                    "some_other_name"
            })
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started.  Var corresponding to
        # sc_hash should not be warm-started.  Var corresponding to sc_vocab
        # should be correctly warm-started after vocab remapping.
        self._assert_cols_to_vars(cols_to_vars, {
            sc_keys:
                np.split(prev_keys_val, 2),
            sc_hash: [np.zeros([8, 1]), np.zeros([7, 1])],
            sc_vocab: [
                np.array([[3.], [2.], [1.]]),
                np.array([[0.5], [0.], [0.]])
            ]
        }, sess)

  def testWarmStartMoreSettingsNoPartitioning(self):
    # [FUNCTION DOC]
    # Scenario: the same as the more-settings test but the model has no partitioner. Expected:
    # key-list weights loaded via the rename, vocabulary weights remapped into one six-row
    # variable (3, 2, 1, 0.5, 0, 0), hashed weights unchanged (zeros).
    # Create old and new vocabs for sparse column "sc_vocab".
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry",
         "blueberry"], "new_vocab")
    # Create feature columns.
    sc_hash = fc.categorical_column_with_hash_bucket(
        "sc_hash", hash_bucket_size=15)
    sc_keys = fc.categorical_column_with_vocabulary_list(
        "sc_keys", vocabulary_list=["a", "b", "c", "e"])
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=new_vocab_path, vocabulary_size=6)
    all_linear_cols = [sc_hash, sc_keys, sc_vocab]

    # Save checkpoint from which to warm-start.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        variable_scope.get_variable(
            "linear_model/sc_hash/weights", shape=[15, 1], initializer=norms())
        sc_keys_weights = variable_scope.get_variable(
            "some_other_name", shape=[4, 1], initializer=rand())
        variable_scope.get_variable(
            "linear_model/sc_vocab/weights",
            initializer=[[0.5], [1.], [2.], [3.]])
        self._write_checkpoint(sess)
        prev_keys_val = self.evaluate(sc_keys_weights)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model(all_linear_cols,
                                                 partitioner=None)
        vocab_info = ws_util.VocabInfo(
            new_vocab=sc_vocab.vocabulary_file,
            new_vocab_size=sc_vocab.vocabulary_size,
            num_oov_buckets=sc_vocab.num_oov_buckets,
            old_vocab=prev_vocab_path)
        ws_util.warm_start(
            self.get_temp_dir(),
            vars_to_warm_start=".*(sc_keys|sc_vocab).*",
            var_name_to_vocab_info={
                ws_util._infer_var_name(cols_to_vars[sc_vocab]): vocab_info
            },
            var_name_to_prev_var_name={
                ws_util._infer_var_name(cols_to_vars[sc_keys]):
                    "some_other_name"
            })
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started.  Var corresponding to
        # sc_hash should not be warm-started.  Var corresponding to sc_vocab
        # should be correctly warm-started after vocab remapping.
        self._assert_cols_to_vars(cols_to_vars, {
            sc_keys: [prev_keys_val],
            sc_hash: [np.zeros([15, 1])],
            sc_vocab: [np.array([[3.], [2.], [1.], [0.5], [0.], [0.]])]
        }, sess)

  def testWarmStartVarsToWarmstartIsNone(self):
    # [FUNCTION DOC]
    # Scenario: partitioned hashed, key-list and vocabulary columns. Action: warm start with the
    # variable selection set to nothing, vocabulary info for the vocabulary variable and a rename
    # for the key-list variable. Expected: only the vocabulary variable is warm-started
    # (remapped); the rename is ignored for selection, so key-list and hashed weights stay zero.
    # Create old and new vocabs for sparse column "sc_vocab".
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry",
         "blueberry"], "new_vocab")
    # Create feature columns.
    sc_hash = fc.categorical_column_with_hash_bucket(
        "sc_hash", hash_bucket_size=15)
    sc_keys = fc.categorical_column_with_vocabulary_list(
        "sc_keys", vocabulary_list=["a", "b", "c", "e"])
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=new_vocab_path, vocabulary_size=6)
    all_linear_cols = [sc_hash, sc_keys, sc_vocab]

    # Save checkpoint from which to warm-start.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        variable_scope.get_variable(
            "linear_model/sc_hash/weights", shape=[15, 1], initializer=norms())
        variable_scope.get_variable(
            "some_other_name", shape=[4, 1], initializer=rand())
        variable_scope.get_variable(
            "linear_model/sc_vocab/weights",
            initializer=[[0.5], [1.], [2.], [3.]])
        self._write_checkpoint(sess)

    def _partitioner(shape, dtype):  # pylint:disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: partitioner for the test's model. Inputs: shape and dtype (dtype unused).
      # Returns a per-dimension partition count list that splits the first dimension in two
      # (or fewer if smaller) and leaves other dimensions whole.
      # Partition each var into 2 equal slices.
      partitions = [1] * len(shape)
      partitions[0] = min(2, shape.dims[0].value)
      return partitions

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = self._create_linear_model(all_linear_cols, _partitioner)
        vocab_info = ws_util.VocabInfo(
            new_vocab=sc_vocab.vocabulary_file,
            new_vocab_size=sc_vocab.vocabulary_size,
            num_oov_buckets=sc_vocab.num_oov_buckets,
            old_vocab=prev_vocab_path)
        ws_util.warm_start(
            self.get_temp_dir(),
            # The special value of None here will ensure that only the variable
            # specified in var_name_to_vocab_info (sc_vocab embedding) is
            # warm-started.
            vars_to_warm_start=None,
            var_name_to_vocab_info={
                ws_util._infer_var_name(cols_to_vars[sc_vocab]): vocab_info
            },
            # Even though this is provided, the None value for
            # vars_to_warm_start overrides the logic, and this will not be
            # warm-started.
            var_name_to_prev_var_name={
                ws_util._infer_var_name(cols_to_vars[sc_keys]):
                    "some_other_name"
            })
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started.  Var corresponding to
        # sc_vocab should be correctly warm-started after vocab remapping,
        # and neither of the other two should be warm-started..
        self._assert_cols_to_vars(cols_to_vars, {
            sc_keys: [np.zeros([2, 1]), np.zeros([2, 1])],
            sc_hash: [np.zeros([8, 1]), np.zeros([7, 1])],
            sc_vocab: [
                np.array([[3.], [2.], [1.]]),
                np.array([[0.5], [0.], [0.]])
            ]
        }, sess)

  def testWarmStartEmbeddingColumn(self):
    # [FUNCTION DOC]
    # Scenario: a checkpoint stores a 4x2 embedding table for a vocabulary column; the new model
    # has an embedding column (dimension two) over a six-word reordered vocabulary, partitioned
    # in two. Vocabulary info uses a constant-valued random initializer (0.42) for new words.
    # Expected: embedding rows are remapped by word and new words filled with 0.42.
    # Create old and new vocabs for embedding column "sc_vocab".
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry", "blueberry"],
        "new_vocab")

    # Save checkpoint from which to warm-start.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        variable_scope.get_variable(
            "input_layer/sc_vocab_embedding/embedding_weights",
            initializer=[[0.5, 0.4], [1., 1.1], [2., 2.2], [3., 3.3]])
        self._write_checkpoint(sess)

    def _partitioner(shape, dtype):  # pylint:disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: partitioner for the test's model. Inputs: shape and dtype (dtype unused).
      # Returns a per-dimension partition count list that splits the first dimension in two
      # (or fewer if smaller) and leaves other dimensions whole.
      # Partition each var into 2 equal slices.
      partitions = [1] * len(shape)
      partitions[0] = min(2, shape.dims[0].value)
      return partitions

    # Create feature columns.
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=new_vocab_path, vocabulary_size=6)
    emb_vocab_column = fc.embedding_column(
        categorical_column=sc_vocab,
        dimension=2)
    all_deep_cols = [emb_vocab_column]
    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = {}
        with variable_scope.variable_scope("", partitioner=_partitioner):
          # Create the variables.
          fc.input_layer(
              features=self._create_dummy_inputs(),
              feature_columns=all_deep_cols,
              cols_to_vars=cols_to_vars)
        vocab_info = ws_util.VocabInfo(
            new_vocab=sc_vocab.vocabulary_file,
            new_vocab_size=sc_vocab.vocabulary_size,
            num_oov_buckets=sc_vocab.num_oov_buckets,
            old_vocab=prev_vocab_path,
            # Can't use constant_initializer with load_and_remap.  In practice,
            # use a truncated normal initializer.
            backup_initializer=init_ops.random_uniform_initializer(
                minval=0.42, maxval=0.42))
        ws_util.warm_start(
            self.get_temp_dir(),
            var_name_to_vocab_info={
                ws_util._infer_var_name(cols_to_vars[emb_vocab_column]):
                    vocab_info
            })
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started. Var corresponding to
        # emb_vocab_column should be correctly warm-started after vocab
        # remapping. Missing values are filled in with the EmbeddingColumn's
        # initializer.
        self._assert_cols_to_vars(
            cols_to_vars, {
                emb_vocab_column: [
                    np.array([[3., 3.3], [2., 2.2], [1., 1.1]]),
                    np.array([[0.5, 0.4], [0.42, 0.42], [0.42, 0.42]])
                ]
            }, sess)

  def testWarmStartEmbeddingColumnLinearModel(self):
    # [FUNCTION DOC]
    # Scenario: like the embedding test but built with a linear model, so the embedding column
    # has both embedding weights and linear weights; the checkpoint stores both. Action: warm
    # start variables matching the vocabulary column with vocabulary info for the embedding table.
    # Expected: linear weights restored (0.69 and 0.71 in two parts), embedding rows remapped by
    # word with 0.42 for new words.
    # Create old and new vocabs for embedding column "sc_vocab".
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry", "blueberry"],
        "new_vocab")

    # Save checkpoint from which to warm-start.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        variable_scope.get_variable(
            "linear_model/sc_vocab_embedding/embedding_weights",
            initializer=[[0.5, 0.4], [1., 1.1], [2., 2.2], [3., 3.3]])
        variable_scope.get_variable(
            "linear_model/sc_vocab_embedding/weights",
            initializer=[[0.69], [0.71]])
        self._write_checkpoint(sess)

    def _partitioner(shape, dtype):  # pylint:disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: partitioner for the test's model. Inputs: shape and dtype (dtype unused).
      # Returns a per-dimension partition count list that splits the first dimension in two
      # (or fewer if smaller) and leaves other dimensions whole.
      # Partition each var into 2 equal slices.
      partitions = [1] * len(shape)
      partitions[0] = min(2, shape.dims[0].value)
      return partitions

    # Create feature columns.
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=new_vocab_path, vocabulary_size=6)
    emb_vocab = fc.embedding_column(
        categorical_column=sc_vocab,
        dimension=2)
    all_deep_cols = [emb_vocab]
    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        cols_to_vars = {}
        with variable_scope.variable_scope("", partitioner=_partitioner):
          # Create the variables.
          fc.linear_model(
              features=self._create_dummy_inputs(),
              feature_columns=all_deep_cols,
              cols_to_vars=cols_to_vars)

        # Construct the vocab_info for the embedding weight.
        vocab_info = ws_util.VocabInfo(
            new_vocab=sc_vocab.vocabulary_file,
            new_vocab_size=sc_vocab.vocabulary_size,
            num_oov_buckets=sc_vocab.num_oov_buckets,
            old_vocab=prev_vocab_path,
            # Can't use constant_initializer with load_and_remap.  In practice,
            # use a truncated normal initializer.
            backup_initializer=init_ops.random_uniform_initializer(
                minval=0.42, maxval=0.42))
        ws_util.warm_start(
            self.get_temp_dir(),
            vars_to_warm_start=".*sc_vocab.*",
            var_name_to_vocab_info={
                "linear_model/sc_vocab_embedding/embedding_weights": vocab_info
            })
        self.evaluate(variables.global_variables_initializer())
        # Verify weights were correctly warm-started. Var corresponding to
        # emb_vocab should be correctly warm-started after vocab remapping.
        # Missing values are filled in with the EmbeddingColumn's initializer.
        self._assert_cols_to_vars(
            cols_to_vars,
            {
                emb_vocab: [
                    # linear weights part 0.
                    np.array([[0.69]]),
                    # linear weights part 1.
                    np.array([[0.71]]),
                    # embedding_weights part 0.
                    np.array([[3., 3.3], [2., 2.2], [1., 1.1]]),
                    # embedding_weights part 1.
                    np.array([[0.5, 0.4], [0.42, 0.42], [0.42, 0.42]])
                ]
            },
            sess)

  def testErrorConditions(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the first variable is created in the default graph outside any explicit
    # graph context, so it leaks into whatever default graph the test runner provides.
    # Scenario: (1) passing a list holding a partitioned variable to the vocabulary helper must
    # raise a type error; (2) after saving a checkpoint, warm start with vocabulary info or a
    # rename for a variable name that does not exist must raise a value error each time.
    x = variable_scope.get_variable(
        "x",
        shape=[4, 1],
        initializer=ones(),
        partitioner=lambda shape, dtype: [2, 1])

    # List of PartitionedVariable is invalid type when warm-starting with vocab.
    self.assertRaises(TypeError, ws_util._warm_start_var_with_vocab, [x],
                      "/tmp", 5, "/tmp", "/tmp")

    # Unused variable names raises ValueError.
    with ops.Graph().as_default():
      with self.cached_session() as sess:
        x = variable_scope.get_variable(
            "x",
            shape=[4, 1],
            initializer=ones(),
            partitioner=lambda shape, dtype: [2, 1])
        self._write_checkpoint(sess)

    self.assertRaises(
        ValueError,
        ws_util.warm_start,
        self.get_temp_dir(),
        var_name_to_vocab_info={"y": ws_util.VocabInfo("", 1, 0, "")})
    self.assertRaises(
        ValueError,
        ws_util.warm_start,
        self.get_temp_dir(),
        var_name_to_prev_var_name={"y": "y2"})

  def testWarmStartFromObjectBasedCheckpoint(self):
    # [FUNCTION DOC]
    # Scenario: an object-based checkpoint is saved from a variable (tracked under the name "v")
    # holding 4x1 values. In a new graph a zero variable with the original variable name exists.
    # Action: warm start from the directory with default settings. Expected: the variable is
    # initialized to the saved values, showing object-based key renaming works.
    prev_val = [[0.5], [1.], [1.5], [2.]]
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        prev_var = variable_scope.get_variable(
            "fruit_weights",
            initializer=prev_val)
        self.evaluate(variables.global_variables_initializer())
        # Save object-based checkpoint.
        tracking_util.Checkpoint(v=prev_var).save(
            os.path.join(self.get_temp_dir(), "checkpoint"))

    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        fruit_weights = variable_scope.get_variable(
            "fruit_weights", initializer=[[0.], [0.], [0.], [0.]])
        ws_util.warm_start(self.get_temp_dir())
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose(prev_val, self.evaluate(fruit_weights))


if __name__ == "__main__":
  test.main()
