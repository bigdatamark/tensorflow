# [MODULE DOC]
# Unit tests for the TPU feature column module (feature_column_v2). Covers the TPU embedding
# column and shared embedding column factories: default and explicit constructor arguments,
# deep copying, variable-scope validation and its bypass, and argument validation for
# device-specific (tensor core) shapes. Defines a small test state manager and a session helper.
# Runs under the standard TensorFlow test runner when executed as a script.
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
# ===================================================================
"""Tests for python.tpu.feature_column."""

import copy

from absl.testing import parameterized

from tensorflow.python.client import session
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.tpu import feature_column_v2 as tpu_fc
from tensorflow.python.tpu import tpu_function


def _initialized_session():
  # [FUNCTION DOC]
  # HELPER: small shared test utility for creating ready-to-use sessions.
  # KNOWN ISSUE: unused in this file; no test calls it.
  # Creates a new session, runs the global variable initializers and the lookup table
  # initializers in it, and returns the session. Takes no inputs.
  sess = session.Session()
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess


class _TestStateManager(fc_lib.StateManager):
  # [CLASS DOC]
  # HELPER: minimal feature column state manager used by the tests below.
  # Keeps a nested mapping from feature column to named variables, creating variables through
  # the variable scope machinery on first request and returning the existing one afterwards.
  # Holds a trainable flag that can disable training of every created variable.

  def __init__(self, trainable=True):
    # [FUNCTION DOC]
    # Constructor. Input: trainable (boolean, default true) that gates whether created variables
    # are trainable. Initializes an empty variable registry. Returns nothing.
    self._all_variables = {}
    self._trainable = trainable

  def create_variable(self,
                      feature_column,
                      name,
                      shape,
                      dtype=None,
                      trainable=True,
                      use_resource=True,
                      initializer=None):
    # [FUNCTION DOC]
    # Returns the variable registered under the given feature column and name, creating it if
    # absent. Inputs: feature_column (the owning column), name (variable name), shape, dtype
    # (optional), trainable (default true; effective only if the manager is also trainable),
    # use_resource (default true) and initializer (optional). New variables are created via the
    # variable scope getter (a side effect on the current graph/variable scope) and recorded.
    if feature_column not in self._all_variables:
      self._all_variables[feature_column] = {}
    var_dict = self._all_variables[feature_column]
    if name in var_dict:
      return var_dict[name]
    else:
      var = variable_scope.get_variable(
          name=name,
          shape=shape,
          dtype=dtype,
          trainable=self._trainable and trainable,
          use_resource=use_resource,
          initializer=initializer)
      var_dict[name] = var
      return var

  def get_variable(self, feature_column, name):
    # [FUNCTION DOC]
    # Looks up a previously created variable. Inputs: feature_column and variable name. Returns
    # the variable. Raises a key error if the column or the name was never registered.
    return self._all_variables[feature_column][name]


class EmbeddingColumnTestV2(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the single TPU embedding column factory and its underlying column class.
  # Covers defaults, explicit arguments, deep copy and scope validation behavior. Uses the
  # test state manager fixture where variable creation is needed.

  def test_defaults(self):
    # [FUNCTION DOC]
    # Scenario: build an embedding column over an identity categorical column with only the
    # dimension given. Asserts the categorical column is the same object, the dimension is kept,
    # the combiner defaults to mean, the name is the key plus an embedding suffix, and the
    # variable shape equals the dimension. The default initializer is not checked (random).
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = tpu_fc.embedding_column_v2(
        categorical_column, dimension=embedding_dimension)
    # Can't test default initializer as it's a random function.
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('mean', embedding_column.combiner)
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual((embedding_dimension,), embedding_column.variable_shape)

  def test_all_constructor_args(self):
    # [FUNCTION DOC]
    # Scenario: build an embedding column passing dimension, a custom combiner name and a custom
    # initializer callable. Asserts all values are exposed unchanged (initializer returns its
    # marker value), the name and variable shape are as expected, and the parse spec is a
    # variable-length int64 feature keyed by the column key.
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = tpu_fc.embedding_column_v2(
        categorical_column,
        dimension=embedding_dimension,
        combiner='my_combiner',
        initializer=lambda: 'my_initializer')
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('my_combiner', embedding_column.combiner)
    self.assertEqual('my_initializer', embedding_column.initializer())
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual((embedding_dimension,), embedding_column.variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column._parse_example_spec)

  def test_deepcopy(self):
    # [FUNCTION DOC]
    # Scenario: deep copy an embedding column. Asserts the copy keeps the same dimension and
    # maximum sequence length as the original.
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_column = tpu_fc.embedding_column_v2(
        categorical_column, dimension=2)
    embedding_column_copy = copy.deepcopy(embedding_column)
    self.assertEqual(embedding_column.dimension,
                     embedding_column_copy.dimension)
    self.assertEqual(embedding_column._max_sequence_length,
                     embedding_column_copy._max_sequence_length)

  def test_with_scope_validation(self):
    # [FUNCTION DOC]
    # Scenario: construct the column directly with scope validation enabled, then inside a
    # one-shard TPU context create its state in one variable scope and again in a different
    # scope. The second creation must raise a value error saying the variable scope name is
    # different. Also asserts the categorical column and dimension are retained.
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    initializer = init_ops.truncated_normal_initializer(mean=0.0, stddev=.5)
    embedding_column = tpu_fc._TPUEmbeddingColumnV2(
        categorical_column=categorical_column,
        dimension=embedding_dimension,
        combiner='mean',
        initializer=initializer,
        max_sequence_length=0,
        learning_rate_fn=None,
        use_safe_embedding_lookup=True,
        bypass_scope_validation=False)
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    state_manager = _TestStateManager()
    with tpu_function.tpu_shard_context(1):
      with variable_scope.variable_scope('tower1/scope1'):
        embedding_column.create_state(state_manager)
      with variable_scope.variable_scope('tower2/scope2'):
        # With default scope validation, the same column cannot be used in a new
        # variable scope.
        with self.assertRaisesRegex(ValueError,
                                    'the variable scope name is different'):
          embedding_column.create_state(state_manager)

  def test_bypass_scope_validation(self):
    # [FUNCTION DOC]
    # Scenario: same as the scope validation test but with validation bypassed. Creating state
    # in two different variable scopes inside a one-shard TPU context must succeed without error.
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    initializer = init_ops.truncated_normal_initializer(mean=0.0, stddev=.5)
    embedding_column = tpu_fc._TPUEmbeddingColumnV2(
        categorical_column=categorical_column,
        dimension=embedding_dimension,
        combiner='mean',
        initializer=initializer,
        max_sequence_length=0,
        learning_rate_fn=None,
        use_safe_embedding_lookup=True,
        bypass_scope_validation=True)
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    state_manager = _TestStateManager()
    with tpu_function.tpu_shard_context(1):
      with variable_scope.variable_scope('tower1/scope1'):
        embedding_column.create_state(state_manager)
      with variable_scope.variable_scope('tower2/scope2'):
        embedding_column.create_state(state_manager)

  def test_deepcopy_with_bypass_scope_validation(self):
    # [FUNCTION DOC]
    # Scenario: deep copy a column built with scope validation bypassed and safe embedding
    # lookup disabled. Asserts the copy preserves dimension, maximum sequence length, the bypass
    # flag (true) and the safe lookup flag (false).
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    initializer = init_ops.truncated_normal_initializer(mean=0.0, stddev=.5)
    embedding_column = tpu_fc._TPUEmbeddingColumnV2(
        categorical_column=categorical_column,
        dimension=embedding_dimension,
        combiner='mean',
        initializer=initializer,
        max_sequence_length=0,
        use_safe_embedding_lookup=False,
        bypass_scope_validation=True)
    embedding_column_copy = copy.deepcopy(embedding_column)
    self.assertEqual(embedding_dimension, embedding_column_copy.dimension)
    self.assertEqual(embedding_column._max_sequence_length,
                     embedding_column_copy._max_sequence_length)
    self.assertTrue(embedding_column_copy._bypass_scope_validation)
    self.assertFalse(embedding_column_copy.use_safe_embedding_lookup)


class SharedEmbeddingColumnTestV2(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the shared embedding columns factory, where several categorical columns share
  # one embedding table. Covers default and explicit arguments and deep copy.

  @test_util.deprecated_graph_mode_only
  def test_defaults(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): create shared embedding columns for two identity categorical
    # columns with only a dimension. Asserts each result maps to its categorical column (order
    # follows the input order), table size is vocabulary by dimension, combiner is mean, an
    # initializer exists, the shared variable name joins the keys, per-column names carry the
    # shared suffix, and the variable shape equals the dimension.
    vocabulary_size = 3
    categorical_column_a = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc_lib.categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_dimension = 2
    embedding_column_b, embedding_column_a = tpu_fc.shared_embedding_columns_v2(
        [categorical_column_b, categorical_column_a],
        dimension=embedding_dimension)
    self.assertIs(categorical_column_a, embedding_column_a.categorical_column)
    self.assertIs(categorical_column_b, embedding_column_b.categorical_column)
    self.assertEqual((vocabulary_size, embedding_dimension),
                     embedding_column_a.get_embedding_table_size())
    self.assertEqual((vocabulary_size, embedding_dimension),
                     embedding_column_a.get_embedding_table_size())
    self.assertEqual('mean', embedding_column_a.combiner)
    self.assertEqual('mean', embedding_column_b.combiner)
    self.assertIsNotNone(embedding_column_a.get_initializer())
    self.assertIsNotNone(embedding_column_b.get_initializer())
    self.assertEqual('aaa_bbb_shared_embedding',
                     embedding_column_a.get_embedding_var_name())
    self.assertEqual('aaa_bbb_shared_embedding',
                     embedding_column_b.get_embedding_var_name())
    self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
    self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
    self.assertEqual((embedding_dimension,), embedding_column_a.variable_shape)
    self.assertEqual((embedding_dimension,), embedding_column_b.variable_shape)

  @test_util.deprecated_graph_mode_only
  def test_all_constructor_args(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the table size assertion is repeated for column a twice and never checks
    # column b, so column b's table size goes untested.
    # Scenario (graph mode only): create shared embedding columns with a custom combiner, a
    # custom initializer and an explicit shared collection name. Asserts the combiner and
    # initializer output are exposed on both columns, the embedding variable name equals the
    # given collection name, and names, table size and variable shape are as expected.
    vocabulary_size = 3
    categorical_column_a = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc_lib.categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_dimension = 2
    embedding_column_a, embedding_column_b = tpu_fc.shared_embedding_columns_v2(
        [categorical_column_a, categorical_column_b],
        dimension=embedding_dimension,
        combiner='my_combiner',
        initializer=lambda: 'my_initializer',
        shared_embedding_collection_name='var_scope_name')
    self.assertIs(categorical_column_a, embedding_column_a.categorical_column)
    self.assertIs(categorical_column_b, embedding_column_b.categorical_column)
    self.assertEqual((vocabulary_size, embedding_dimension),
                     embedding_column_a.get_embedding_table_size())
    self.assertEqual((vocabulary_size, embedding_dimension),
                     embedding_column_a.get_embedding_table_size())
    self.assertEqual('my_combiner', embedding_column_a.combiner)
    self.assertEqual('my_combiner', embedding_column_b.combiner)
    self.assertEqual('my_initializer', embedding_column_a.get_initializer()())
    self.assertEqual('my_initializer', embedding_column_b.get_initializer()())
    self.assertEqual('var_scope_name',
                     embedding_column_a.get_embedding_var_name())
    self.assertEqual('var_scope_name',
                     embedding_column_b.get_embedding_var_name())
    self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
    self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
    self.assertEqual((embedding_dimension,), embedding_column_a.variable_shape)
    self.assertEqual((embedding_dimension,), embedding_column_b.variable_shape)

  def test_deepcopy(self):
    # [FUNCTION DOC]
    # Scenario: deep copy the list of shared embedding columns. Asserts the shared embedding
    # collection name of every copied column equals that of the corresponding original.
    vocabulary_size = 3
    categorical_column_a = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc_lib.categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_dimension = 2
    columns = tpu_fc.shared_embedding_columns_v2(
        [categorical_column_b, categorical_column_a],
        dimension=embedding_dimension)
    columns_copy = copy.deepcopy(columns)
    self.assertEqual(
        [column._shared_embedding_collection_name for column in columns],
        [column._shared_embedding_collection_name for column in columns_copy])


class DeviceSpecificEmbeddingColumnTestV2(test.TestCase,
                                          parameterized.TestCase):
  # [CLASS DOC]
  # Test case for argument validation of device-specific (tensor core) shape options when
  # creating shared embedding columns.

  @test_util.deprecated_graph_mode_only
  def test_error_dense_shape_invalid(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): request shared embedding columns with a tensor core shape of
    # three entries. Expects a value error stating the shape must be of size 2.
    categorical_column_input = fc_lib.categorical_column_with_identity(
        key='inp', num_buckets=5)
    with self.assertRaisesRegex(ValueError, 'tensor_core_shape must be size 2'):
      tpu_fc.shared_embedding_columns_v2([categorical_column_input],
                                         dimension=20,
                                         tensor_core_shape=[None, 20, 15])


if __name__ == '__main__':
  test.main()
