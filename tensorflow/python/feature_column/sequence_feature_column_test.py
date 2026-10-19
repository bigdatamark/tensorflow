# [MODULE DOC]
# Unit tests for the sequence feature column module. Covers the context-and-sequence concatenation
# utility, the four sequence categorical column factories (identity, hash bucket, vocabulary file,
# vocabulary list) including two-dimensional and three-dimensional sparse inputs, sequence-aware
# shared embedding, indicator and numeric columns, sequence length computation (including empty
# rows), argument validation and serialization. Most classes run in both graph and eager modes.
# Contains small shared helpers for session setup, sparse tensor comparison and fetching dense or
# sparse outputs from columns. No import-time side effects; runs via the test main entry point.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sequential_feature_column."""

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import sequence_feature_column as sfc
from tensorflow.python.feature_column import serialization
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


def _initialized_session(config=None):
  # [FUNCTION DOC]
  # HELPER: shared test utility. Creates a new session with the optional configuration, runs the
  # global variable initializer and the lookup table initializer, and returns the session. The
  # caller owns and should close it.
  sess = session.Session(config=config)
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess


@test_util.run_all_in_graph_and_eager_modes
class ConcatenateContextInputTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the concatenate-context-input utility: correct tiling and concatenation, plus the errors
  # for wrong ranks and wrong dtypes. Runs in graph and eager modes and is parameterized.
  """Tests the utility fn concatenate_context_input."""

  def test_concatenate_context_input(self):
    # [FUNCTION DOC]
    # Scenario: a float sequence tensor of shape two by three by two and a float context tensor of
    # shape two by five. Expects the output to be the sequence with each example's context row
    # repeated at every timestep and appended on the last axis (shape two by three by seven).
    seq_input = ops.convert_to_tensor(np.arange(12).reshape(2, 3, 2))
    context_input = ops.convert_to_tensor(np.arange(10).reshape(2, 5))
    seq_input = math_ops.cast(seq_input, dtype=dtypes.float32)
    context_input = math_ops.cast(context_input, dtype=dtypes.float32)
    input_layer = sfc.concatenate_context_input(context_input, seq_input)

    expected = np.array([
        [[0, 1, 0, 1, 2, 3, 4], [2, 3, 0, 1, 2, 3, 4], [4, 5, 0, 1, 2, 3, 4]],
        [[6, 7, 5, 6, 7, 8, 9], [8, 9, 5, 6, 7, 8, 9], [10, 11, 5, 6, 7, 8, 9]]
    ], dtype=np.float32)
    output = self.evaluate(input_layer)
    self.assertAllEqual(expected, output)

  @parameterized.named_parameters(
      {'testcase_name': 'rank_lt_3',
       'seq_input_arg': np.arange(100).reshape(10, 10)},
      {'testcase_name': 'rank_gt_3',
       'seq_input_arg': np.arange(100).reshape(5, 5, 2, 2)}
      )
  def test_sequence_input_throws_error(self, seq_input_arg):
    # [FUNCTION DOC]
    # Parameterized scenario (sequence input of rank two, and of rank four). Given a valid float
    # context tensor, expects a value error stating that the sequence input must have rank three.
    seq_input = ops.convert_to_tensor(seq_input_arg)
    context_input = ops.convert_to_tensor(np.arange(100).reshape(10, 10))
    seq_input = math_ops.cast(seq_input, dtype=dtypes.float32)
    context_input = math_ops.cast(context_input, dtype=dtypes.float32)
    with self.assertRaisesRegex(ValueError, 'sequence_input must have rank 3'):
      sfc.concatenate_context_input(context_input, seq_input)

  @parameterized.named_parameters(
      {'testcase_name': 'rank_lt_2',
       'context_input_arg': np.arange(100)},
      {'testcase_name': 'rank_gt_2',
       'context_input_arg': np.arange(100).reshape(5, 5, 4)}
      )
  def test_context_input_throws_error(self, context_input_arg):
    # [FUNCTION DOC]
    # Parameterized scenario (context input of rank one, and of rank three) with a valid rank three
    # float sequence tensor. Expects a value error stating that the context input must have rank two.
    context_input = ops.convert_to_tensor(context_input_arg)
    seq_input = ops.convert_to_tensor(np.arange(100).reshape(5, 5, 4))
    seq_input = math_ops.cast(seq_input, dtype=dtypes.float32)
    context_input = math_ops.cast(context_input, dtype=dtypes.float32)
    with self.assertRaisesRegex(ValueError, 'context_input must have rank 2'):
      sfc.concatenate_context_input(context_input, seq_input)

  def test_integer_seq_input_throws_error(self):
    # [FUNCTION DOC]
    # Scenario: an integer-typed sequence tensor with a float context tensor. Expects a type error
    # saying the sequence input must have dtype float32.
    seq_input = ops.convert_to_tensor(np.arange(100).reshape(5, 5, 4))
    context_input = ops.convert_to_tensor(np.arange(100).reshape(10, 10))
    context_input = math_ops.cast(context_input, dtype=dtypes.float32)
    with self.assertRaisesRegex(TypeError,
                                'sequence_input must have dtype float32'):
      sfc.concatenate_context_input(context_input, seq_input)

  def test_integer_context_input_throws_error(self):
    # [FUNCTION DOC]
    # Scenario: a float sequence tensor with an integer-typed context tensor. Expects a type error
    # saying the context input must have dtype float32.
    seq_input = ops.convert_to_tensor(np.arange(100).reshape(5, 5, 4))
    context_input = ops.convert_to_tensor(np.arange(100).reshape(10, 10))
    seq_input = math_ops.cast(seq_input, dtype=dtypes.float32)
    with self.assertRaisesRegex(TypeError,
                                'context_input must have dtype float32'):
      sfc.concatenate_context_input(context_input, seq_input)


def _assert_sparse_tensor_value(test_case, expected, actual):
  # [FUNCTION DOC]
  # HELPER: shared assertion utility. Inputs: a test case, the expected and the actual sparse
  # tensor values. Checks indices and dense shape (via the indices-and-shape helper), then that the
  # values have the same dtype and the same contents. Fails the test on any mismatch.
  _assert_sparse_tensor_indices_shape(test_case, expected, actual)

  test_case.assertEqual(
      np.array(expected.values).dtype, np.array(actual.values).dtype)
  test_case.assertAllEqual(expected.values, actual.values)


def _assert_sparse_tensor_indices_shape(test_case, expected, actual):
  # [FUNCTION DOC]
  # HELPER: shared assertion utility. Checks that the actual indices and dense shape have 64-bit
  # integer dtype and equal the expected ones; values are not compared (useful when values depend
  # on hashing). Fails the test on mismatch.
  test_case.assertEqual(np.int64, np.array(actual.indices).dtype)
  test_case.assertAllEqual(expected.indices, actual.indices)

  test_case.assertEqual(np.int64, np.array(actual.dense_shape).dtype)
  test_case.assertAllEqual(expected.dense_shape, actual.dense_shape)


def _get_sequence_dense_tensor(column, features):
  # [FUNCTION DOC]
  # HELPER: shared test utility. Wraps the features mapping in a transformation cache and returns
  # the pair of dense tensor and sequence length produced by the column, passing no state manager.
  return column.get_sequence_dense_tensor(
      fc.FeatureTransformationCache(features), None)


def _get_sparse_tensors(column, features):
  # [FUNCTION DOC]
  # HELPER: shared test utility. Wraps the features mapping in a transformation cache and returns
  # the pair of id tensor and weight tensor produced by the categorical column, passing no state
  # manager.
  return column.get_sparse_tensors(
      fc.FeatureTransformationCache(features), None)


@test_util.run_all_in_graph_and_eager_modes
class SequenceCategoricalColumnWithIdentityTest(
    test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the sequence categorical column with identity ids: id extraction from two- and
  # three-dimensional sparse input, and config round trips when wrapped by an indicator column.

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': (1, 2, 0),
           'dense_shape': (2, 2)},
       'expected_args': {
           'indices': ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
           'values': np.array((1, 2, 0), dtype=np.int64),
           'dense_shape': (2, 2, 1)}},
      {'testcase_name': '3D',
       'inputs_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': (6, 7, 8),
           'dense_shape': (2, 2, 2)},
       'expected_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': np.array((6, 7, 8), dtype=np.int64),
           'dense_shape': (2, 2, 2)}}
      )
  def test_get_sparse_tensors(self, inputs_args, expected_args):
    # [FUNCTION DOC]
    # Parameterized scenario (two-dimensional and three-dimensional sparse integer input, nine
    # buckets). Expects no weight tensor and an id tensor equal to the input values as 64-bit
    # integers; two-dimensional input gains a trailing dimension of size one, three-dimensional
    # input keeps its indices and shape.
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    expected = sparse_tensor.SparseTensorValue(**expected_args)
    column = sfc.sequence_categorical_column_with_identity('aaa', num_buckets=9)

    id_weight_pair = _get_sparse_tensors(column, {'aaa': inputs})

    self.assertIsNone(id_weight_pair.weight_tensor)
    _assert_sparse_tensor_value(
        self, expected, self.evaluate(id_weight_pair.id_tensor))

  def test_serialization(self):
    # [FUNCTION DOC]
    # Scenario: an indicator column wrapping a sequence identity column. Expects the config to nest
    # the class names and the identity settings (key, bucket count, no default), that rebuilding
    # from the config gives an equal indicator column with a new inner column, and that supplying a
    # columns-by-name mapping makes the rebuilt column reuse the given inner column object.
    """Tests that column can be serialized."""
    parent = sfc.sequence_categorical_column_with_identity(
        'animal', num_buckets=4)
    animal = fc.indicator_column(parent)

    config = animal.get_config()
    self.assertEqual(
        {
            'categorical_column': {
                'class_name': 'SequenceCategoricalColumn',
                'config': {
                    'categorical_column': {
                        'class_name': 'IdentityCategoricalColumn',
                        'config': {
                            'default_value': None,
                            'key': 'animal',
                            'number_buckets': 4
                        }
                    }
                }
            }
        }, config)

    new_animal = fc.IndicatorColumn.from_config(config)
    self.assertEqual(animal, new_animal)
    self.assertIsNot(parent, new_animal.categorical_column)

    new_animal = fc.IndicatorColumn.from_config(
        config,
        columns_by_name={
            serialization._column_name_with_class_name(parent): parent
        })
    self.assertEqual(animal, new_animal)
    self.assertIs(parent, new_animal.categorical_column)


@test_util.run_all_in_graph_and_eager_modes
class SequenceCategoricalColumnWithHashBucketTest(
    test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the sequence categorical column with hashed ids, checking index and shape handling
  # for two- and three-dimensional string input (hash values themselves are not asserted).

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': ('omar', 'stringer', 'marlo'),
           'dense_shape': (2, 2)},
       'expected_args': {
           'indices': ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
           # Ignored to avoid hash dependence in test.
           'values': np.array((0, 0, 0), dtype=np.int64),
           'dense_shape': (2, 2, 1)}},
      {'testcase_name': '3D',
       'inputs_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': ('omar', 'stringer', 'marlo'),
           'dense_shape': (2, 2, 2)},
       'expected_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           # Ignored to avoid hash dependence in test.
           'values': np.array((0, 0, 0), dtype=np.int64),
           'dense_shape': (2, 2, 2)}}
      )
  def test_get_sparse_tensors(self, inputs_args, expected_args):
    # [FUNCTION DOC]
    # Parameterized scenario (two- and three-dimensional string sparse input, ten buckets). Expects
    # no weight tensor and an id tensor whose indices and dense shape match the expected ones;
    # values are intentionally not compared to avoid depending on the hash function.
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    expected = sparse_tensor.SparseTensorValue(**expected_args)
    column = sfc.sequence_categorical_column_with_hash_bucket(
        'aaa', hash_bucket_size=10)

    id_weight_pair = _get_sparse_tensors(column, {'aaa': inputs})

    self.assertIsNone(id_weight_pair.weight_tensor)
    _assert_sparse_tensor_indices_shape(
        self, expected, self.evaluate(id_weight_pair.id_tensor))


@test_util.run_all_in_graph_and_eager_modes
class SequenceCategoricalColumnWithVocabularyFileTest(
    test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the sequence categorical column backed by a vocabulary file. The fixture writes a
  # three word vocabulary file into the test temp directory before each test.

  def _write_vocab(self, vocab_strings, file_name):
    # [FUNCTION DOC]
    # HELPER: fixture utility. Inputs: a list of vocabulary strings and a file name. Writes the
    # strings, one per line, into a file of that name in the test temp directory and returns the
    # file path. Side effect: creates (or overwrites) that file.
    vocab_file = os.path.join(self.get_temp_dir(), file_name)
    with open(vocab_file, 'w') as f:
      f.write('\n'.join(vocab_strings))
    return vocab_file

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: calls the parent setup, writes a vocabulary file with the words omar, stringer
    # and marlo, and stores its path and the size three on the instance for the tests.
    super(SequenceCategoricalColumnWithVocabularyFileTest, self).setUp()

    vocab_strings = ['omar', 'stringer', 'marlo']
    self._wire_vocabulary_file_name = self._write_vocab(vocab_strings,
                                                        'wire_vocabulary.txt')
    self._wire_vocabulary_size = 3

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': ('marlo', 'skywalker', 'omar'),
           'dense_shape': (2, 2)},
       'expected_args': {
           'indices': ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
           'values': np.array((2, -1, 0), dtype=np.int64),
           'dense_shape': (2, 2, 1)}},
      {'testcase_name': '3D',
       'inputs_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': ('omar', 'skywalker', 'marlo'),
           'dense_shape': (2, 2, 2)},
       'expected_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': np.array((0, -1, 2), dtype=np.int64),
           'dense_shape': (2, 2, 2)}}
      )
  def test_get_sparse_tensors(self, inputs_args, expected_args):
    # [FUNCTION DOC]
    # Parameterized scenario (two- and three-dimensional string sparse input, vocabulary file of
    # three words). Initializes variables and lookup tables, then expects no weight tensor and ids
    # equal to the word positions, with minus one for the unknown word; two-dimensional input gains
    # a trailing dimension of size one.
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    expected = sparse_tensor.SparseTensorValue(**expected_args)
    column = sfc.sequence_categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)

    id_weight_pair = _get_sparse_tensors(column, {'aaa': inputs})

    self.assertIsNone(id_weight_pair.weight_tensor)
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())
    _assert_sparse_tensor_value(
        self, expected, self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_dynamic_zero_length(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph, an empty string sparse input with batch two and zero length,
    # fed through a placeholder whose sequence dimension is unknown at graph build time. Runs in an
    # initialized session and expects an empty id tensor of shape two by zero by one, no weights.
    """Tests _get_sparse_tensors with a dynamic sequence length."""
    with ops.Graph().as_default():
      inputs = sparse_tensor.SparseTensorValue(
          indices=np.zeros((0, 2)), values=[], dense_shape=(2, 0))
      expected = sparse_tensor.SparseTensorValue(
          indices=np.zeros((0, 3)),
          values=np.array((), dtype=np.int64),
          dense_shape=(2, 0, 1))
      column = sfc.sequence_categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=self._wire_vocabulary_size)
      input_placeholder_shape = list(inputs.dense_shape)
      # Make second dimension (sequence length) dynamic.
      input_placeholder_shape[1] = None
      input_placeholder = array_ops.sparse_placeholder(
          dtypes.string, shape=input_placeholder_shape)
      id_weight_pair = _get_sparse_tensors(column, {'aaa': input_placeholder})

      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session() as sess:
        result = id_weight_pair.id_tensor.eval(
            session=sess, feed_dict={input_placeholder: inputs})
        _assert_sparse_tensor_value(
            self, expected, result)


@test_util.run_all_in_graph_and_eager_modes
class SequenceCategoricalColumnWithVocabularyListTest(
    test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the sequence categorical column backed by an in-memory vocabulary list.

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': ('marlo', 'skywalker', 'omar'),
           'dense_shape': (2, 2)},
       'expected_args': {
           'indices': ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
           'values': np.array((2, -1, 0), dtype=np.int64),
           'dense_shape': (2, 2, 1)}},
      {'testcase_name': '3D',
       'inputs_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': ('omar', 'skywalker', 'marlo'),
           'dense_shape': (2, 2, 2)},
       'expected_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': np.array((0, -1, 2), dtype=np.int64),
           'dense_shape': (2, 2, 2)}}
      )
  def test_get_sparse_tensors(self, inputs_args, expected_args):
    # [FUNCTION DOC]
    # Parameterized scenario (two- and three-dimensional string sparse input, vocabulary of three
    # words). Initializes variables and lookup tables, then expects no weight tensor and ids equal
    # to the word positions, with minus one for the unknown word; two-dimensional input gains a
    # trailing dimension of size one.
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    expected = sparse_tensor.SparseTensorValue(**expected_args)
    column = sfc.sequence_categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'))

    id_weight_pair = _get_sparse_tensors(column, {'aaa': inputs})

    self.assertIsNone(id_weight_pair.weight_tensor)
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())
    _assert_sparse_tensor_value(
        self, expected, self.evaluate(id_weight_pair.id_tensor))


class SequenceSharedEmbeddingColumnTest(test.TestCase):
  # [CLASS DOC]
  # Tests for shared embedding columns built over sequence identity columns: embedding lookups per
  # timestep and sequence lengths, including examples without ids. Runs in a fresh graph per test.

  def test_get_sequence_dense_tensor(self):
    # [FUNCTION DOC]
    # Scenario: two sequence identity columns share one embedding with vocabulary three and
    # dimension two, using a custom initializer that returns fixed values and asserts the requested
    # shape, dtype and absence of partitioning. Expects exactly one shared variable with the
    # expected name and values, and per-column lookups that pad missing timesteps and empty
    # examples with zeros.
    vocabulary_size = 3
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
        (7., 11.)  # id 2
    )

    def _initializer(shape, dtype, partition_info=None):
      # [FUNCTION DOC]
      # HELPER: embedding initializer for the test above. Asserts the requested shape equals the
      # vocabulary size by embedding dimension, the dtype is float32 and no partition info is given;
      # returns the fixed embedding values.
      self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return embedding_values

    with ops.Graph().as_default():
      sparse_input_a = sparse_tensor.SparseTensorValue(
          # example 0, ids [2]
          # example 1, ids [0, 1]
          # example 2, ids []
          # example 3, ids [1]
          indices=((0, 0), (1, 0), (1, 1), (3, 0)),
          values=(2, 0, 1, 1),
          dense_shape=(4, 2))
      sparse_input_b = sparse_tensor.SparseTensorValue(
          # example 0, ids [1]
          # example 1, ids [0, 2]
          # example 2, ids [0]
          # example 3, ids []
          indices=((0, 0), (1, 0), (1, 1), (2, 0)),
          values=(1, 0, 2, 0),
          dense_shape=(4, 2))

      expected_lookups_a = [
          # example 0, ids [2]
          [[7., 11.], [0., 0.]],
          # example 1, ids [0, 1]
          [[1., 2.], [3., 5.]],
          # example 2, ids []
          [[0., 0.], [0., 0.]],
          # example 3, ids [1]
          [[3., 5.], [0., 0.]],
      ]

      expected_lookups_b = [
          # example 0, ids [1]
          [[3., 5.], [0., 0.]],
          # example 1, ids [0, 2]
          [[1., 2.], [7., 11.]],
          # example 2, ids [0]
          [[1., 2.], [0., 0.]],
          # example 3, ids []
          [[0., 0.], [0., 0.]],
      ]

      categorical_column_a = sfc.sequence_categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      categorical_column_b = sfc.sequence_categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)
      shared_embedding_columns = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          initializer=_initializer)

      embedding_lookup_a = _get_sequence_dense_tensor(
          shared_embedding_columns[0], {'aaa': sparse_input_a})[0]
      embedding_lookup_b = _get_sequence_dense_tensor(
          shared_embedding_columns[1], {'bbb': sparse_input_b})[0]

      self.evaluate(variables_lib.global_variables_initializer())
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('aaa_bbb_shared_embedding:0',),
                            tuple([v.name for v in global_vars]))
      self.assertAllEqual(embedding_values, self.evaluate(global_vars[0]))
      self.assertAllEqual(
          expected_lookups_a, self.evaluate(embedding_lookup_a))
      self.assertAllEqual(expected_lookups_b, self.evaluate(embedding_lookup_b))

  def test_sequence_length(self):
    # [FUNCTION DOC]
    # Scenario: two sequence identity columns share an embedding; each gets a small sparse input
    # with lengths one and two (or two and one). Expects the computed sequence lengths to match and
    # to have 64-bit integer dtype, evaluated in an initialized session.
    with ops.Graph().as_default():
      vocabulary_size = 3

      sparse_input_a = sparse_tensor.SparseTensorValue(
          # example 0, ids [2]
          # example 1, ids [0, 1]
          indices=((0, 0), (1, 0), (1, 1)),
          values=(2, 0, 1),
          dense_shape=(2, 2))
      expected_sequence_length_a = [1, 2]
      categorical_column_a = sfc.sequence_categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)

      sparse_input_b = sparse_tensor.SparseTensorValue(
          # example 0, ids [0, 2]
          # example 1, ids [1]
          indices=((0, 0), (0, 1), (1, 0)),
          values=(0, 2, 1),
          dense_shape=(2, 2))
      expected_sequence_length_b = [2, 1]
      categorical_column_b = sfc.sequence_categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)
      shared_embedding_columns = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b], dimension=2)

      sequence_length_a = _get_sequence_dense_tensor(
          shared_embedding_columns[0], {'aaa': sparse_input_a})[1]
      sequence_length_b = _get_sequence_dense_tensor(
          shared_embedding_columns[1], {'bbb': sparse_input_b})[1]

      with _initialized_session() as sess:
        sequence_length_a = sess.run(sequence_length_a)
        self.assertAllEqual(expected_sequence_length_a, sequence_length_a)
        self.assertEqual(np.int64, sequence_length_a.dtype)
        sequence_length_b = sess.run(sequence_length_b)
        self.assertAllEqual(expected_sequence_length_b, sequence_length_b)
        self.assertEqual(np.int64, sequence_length_b.dtype)

  def test_sequence_length_with_empty_rows(self):
    # [FUNCTION DOC]
    # Scenario: six-example sparse inputs for two shared-embedding sequence columns where some
    # examples (leading, middle and trailing) have no ids. Expects the sequence lengths to be zero
    # for those examples and the id counts for the others.
    """Tests _sequence_length when some examples do not have ids."""
    with ops.Graph().as_default():
      vocabulary_size = 3
      sparse_input_a = sparse_tensor.SparseTensorValue(
          # example 0, ids []
          # example 1, ids [2]
          # example 2, ids [0, 1]
          # example 3, ids []
          # example 4, ids [1]
          # example 5, ids []
          indices=((1, 0), (2, 0), (2, 1), (4, 0)),
          values=(2, 0, 1, 1),
          dense_shape=(6, 2))
      expected_sequence_length_a = [0, 1, 2, 0, 1, 0]
      categorical_column_a = sfc.sequence_categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)

      sparse_input_b = sparse_tensor.SparseTensorValue(
          # example 0, ids [2]
          # example 1, ids []
          # example 2, ids []
          # example 3, ids []
          # example 4, ids [1]
          # example 5, ids [0, 1]
          indices=((0, 0), (4, 0), (5, 0), (5, 1)),
          values=(2, 1, 0, 1),
          dense_shape=(6, 2))
      expected_sequence_length_b = [1, 0, 0, 0, 1, 2]
      categorical_column_b = sfc.sequence_categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)

      shared_embedding_columns = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b], dimension=2)

      sequence_length_a = _get_sequence_dense_tensor(
          shared_embedding_columns[0], {'aaa': sparse_input_a})[1]
      sequence_length_b = _get_sequence_dense_tensor(
          shared_embedding_columns[1], {'bbb': sparse_input_b})[1]

      with _initialized_session() as sess:
        self.assertAllEqual(
            expected_sequence_length_a, sequence_length_a.eval(session=sess))
        self.assertAllEqual(
            expected_sequence_length_b, sequence_length_b.eval(session=sess))


@test_util.run_all_in_graph_and_eager_modes
class SequenceIndicatorColumnTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for indicator columns wrapping sequence identity columns: multi-hot dense output for
  # two- and three-dimensional input and sequence lengths. Runs in graph and eager modes and is
  # parameterized.

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           # example 0, ids [2]
           # example 1, ids [0, 1]
           # example 2, ids []
           # example 3, ids [1]
           'indices': ((0, 0), (1, 0), (1, 1), (3, 0)),
           'values': (2, 0, 1, 1),
           'dense_shape': (4, 2)},
       'expected': [
           # example 0, ids [2]
           [[0., 0., 1.], [0., 0., 0.]],
           # example 1, ids [0, 1]
           [[1., 0., 0.], [0., 1., 0.]],
           # example 2, ids []
           [[0., 0., 0.], [0., 0., 0.]],
           # example 3, ids [1]
           [[0., 1., 0.], [0., 0., 0.]]]},
      {'testcase_name': '3D',
       'inputs_args': {
           # example 0, ids [[2]]
           # example 1, ids [[0, 1], [2]]
           # example 2, ids []
           # example 3, ids [[1], [2, 2]]
           'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0),
                       (3, 0, 0), (3, 1, 0), (3, 1, 1)),
           'values': (2, 0, 1, 2, 1, 2, 2),
           'dense_shape': (4, 2, 2)},
       'expected': [
           # example 0, ids [[2]]
           [[0., 0., 1.], [0., 0., 0.]],
           # example 1, ids [[0, 1], [2]]
           [[1., 1., 0.], [0., 0., 1.]],
           # example 2, ids []
           [[0., 0., 0.], [0., 0., 0.]],
           # example 3, ids [[1], [2, 2]]
           [[0., 1., 0.], [0., 0., 2.]]]}
      )
  def test_get_sequence_dense_tensor(self, inputs_args, expected):
    # [FUNCTION DOC]
    # Parameterized scenario (two-dimensional and three-dimensional sparse input, three buckets).
    # Expects a dense tensor with one indicator vector per timestep, counting repeated ids in the
    # three-dimensional case, and all zeros for examples without ids.
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    vocabulary_size = 3

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    indicator_column = fc.indicator_column(categorical_column)

    indicator_tensor, _ = _get_sequence_dense_tensor(
        indicator_column, {'aaa': inputs})

    self.assertAllEqual(expected, self.evaluate(indicator_tensor))

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           # example 0, ids [2]
           # example 1, ids [0, 1]
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': (2, 0, 1),
           'dense_shape': (2, 2)},
       'expected_sequence_length': [1, 2]},
      {'testcase_name': '3D',
       'inputs_args': {
           # example 0, ids [[2]]
           # example 1, ids [[0, 1], [2]]
           'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
           'values': (2, 0, 1, 2),
           'dense_shape': (2, 2, 2)},
       'expected_sequence_length': [1, 2]}
      )
  def test_sequence_length(self, inputs_args, expected_sequence_length):
    # [FUNCTION DOC]
    # Parameterized scenario (two- and three-dimensional input). Expects sequence lengths of one
    # and two with 64-bit integer dtype.
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    vocabulary_size = 3

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    indicator_column = fc.indicator_column(categorical_column)

    _, sequence_length = _get_sequence_dense_tensor(
        indicator_column, {'aaa': inputs})

    sequence_length = self.evaluate(sequence_length)
    self.assertAllEqual(expected_sequence_length, sequence_length)
    self.assertEqual(np.int64, sequence_length.dtype)

  def test_sequence_length_with_empty_rows(self):
    # [FUNCTION DOC]
    # Scenario: six examples where some have no ids. Expects zero sequence length for the empty
    # examples and the id counts for the others.
    """Tests _sequence_length when some examples do not have ids."""
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids []
        # example 1, ids [2]
        # example 2, ids [0, 1]
        # example 3, ids []
        # example 4, ids [1]
        # example 5, ids []
        indices=((1, 0), (2, 0), (2, 1), (4, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(6, 2))
    expected_sequence_length = [0, 1, 2, 0, 1, 0]

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    indicator_column = fc.indicator_column(categorical_column)

    _, sequence_length = _get_sequence_dense_tensor(
        indicator_column, {'aaa': sparse_input})

    self.assertAllEqual(
        expected_sequence_length, self.evaluate(sequence_length))


@test_util.run_all_in_graph_and_eager_modes
class SequenceNumericColumnTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the sequence numeric column: constructor defaults and validation, dense output for
  # various shapes and input ranks, normalizer application, sequence lengths, serialization and
  # parents. Runs in graph and eager modes and is parameterized.

  def test_defaults(self):
    # [FUNCTION DOC]
    # Scenario: a column created with only a key. Expects key and name to equal the key, shape of
    # one, default value zero, float32 dtype and no normalizer.
    a = sfc.sequence_numeric_column('aaa')
    self.assertEqual('aaa', a.key)
    self.assertEqual('aaa', a.name)
    self.assertEqual((1,), a.shape)
    self.assertEqual(0., a.default_value)
    self.assertEqual(dtypes.float32, a.dtype)
    self.assertIsNone(a.normalizer_fn)

  def test_shape_saved_as_tuple(self):
    # [FUNCTION DOC]
    # Scenario: shape passed as a list. Expects the stored shape to be an equal tuple.
    a = sfc.sequence_numeric_column('aaa', shape=[1, 2])
    self.assertEqual((1, 2), a.shape)

  def test_shape_must_be_positive_integer(self):
    # [FUNCTION DOC]
    # Scenario: invalid shapes. A floating dimension must raise a type error about integer
    # dimensions; a zero dimension must raise a value error about dimensions greater than zero.
    with self.assertRaisesRegex(TypeError, 'shape dimensions must be integer'):
      sfc.sequence_numeric_column('aaa', shape=[1.0])

    with self.assertRaisesRegex(ValueError,
                                'shape dimensions must be greater than 0'):
      sfc.sequence_numeric_column('aaa', shape=[0])

  def test_dtype_is_convertible_to_float(self):
    # [FUNCTION DOC]
    # Scenario: string dtype. Expects a value error saying the dtype must be convertible to float.
    with self.assertRaisesRegex(ValueError,
                                'dtype must be convertible to float'):
      sfc.sequence_numeric_column('aaa', dtype=dtypes.string)

  def test_normalizer_fn_must_be_callable(self):
    # [FUNCTION DOC]
    # Scenario: a string given as the normalizer. Expects a type error saying it must be callable.
    with self.assertRaisesRegex(TypeError, 'must be a callable'):
      sfc.sequence_numeric_column('aaa', normalizer_fn='NotACallable')

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           # example 0, values [0., 1]
           # example 1, [10.]
           'indices': ((0, 0), (0, 1), (1, 0)),
           'values': (0., 1., 10.),
           'dense_shape': (2, 2)},
       'expected': [
           [[0.], [1.]],
           [[10.], [0.]]]},
      {'testcase_name': '3D',
       'inputs_args': {
           # feature 0, ids [[20, 3], [5]]
           # feature 1, ids [[3], [8]]
           'indices': ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)),
           'values': (20, 3, 5., 3., 8.),
           'dense_shape': (2, 2, 2)},
       'expected': [
           [[20.], [3.], [5.], [0.]],
           [[3.], [0.], [8.], [0.]]]},
      )
  def test_get_sequence_dense_tensor(self, inputs_args, expected):
    # [FUNCTION DOC]
    # Parameterized scenario (two- and three-dimensional sparse float input, default shape).
    # Expects the dense tensor to hold one value per timestep, padded with zeros; in the
    # three-dimensional case the inner groups are flattened into extra timesteps.
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    numeric_column = sfc.sequence_numeric_column('aaa')

    dense_tensor, _ = _get_sequence_dense_tensor(
        numeric_column, {'aaa': inputs})
    self.assertAllEqual(expected, self.evaluate(dense_tensor))

  def test_get_sequence_dense_tensor_with_normalizer_fn(self):
    # [FUNCTION DOC]
    # Scenario: a normalizer that adds two to the entries at positions zero-zero and one-one using
    # a sparse addition. Expects the dense output to reflect the added values, including at a
    # position that was previously empty.

    def _increment_two(input_sparse_tensor):
      # [FUNCTION DOC]
      # HELPER: normalizer for the test above. Takes a sparse tensor and returns its sparse sum with
      # a fixed sparse tensor holding two at positions zero-zero and one-one (shape two by two).
      return sparse_ops.sparse_add(
          input_sparse_tensor,
          sparse_tensor.SparseTensor(((0, 0), (1, 1)), (2.0, 2.0), (2, 2))
      )

    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, values [[0.], [1]]
        # example 1, [[10.]]
        indices=((0, 0), (0, 1), (1, 0)),
        values=(0., 1., 10.),
        dense_shape=(2, 2))

    # Before _increment_two:
    #   [[0.], [1.]],
    #   [[10.], [0.]],
    # After _increment_two:
    #   [[2.], [1.]],
    #   [[10.], [2.]],
    expected_dense_tensor = [
        [[2.], [1.]],
        [[10.], [2.]],
    ]
    numeric_column = sfc.sequence_numeric_column(
        'aaa', normalizer_fn=_increment_two)

    dense_tensor, _ = _get_sequence_dense_tensor(
        numeric_column, {'aaa': sparse_input})

    self.assertAllEqual(
        expected_dense_tensor, self.evaluate(dense_tensor))

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'sparse_input_args': {
           # example 0, values [[[0., 1.],  [2., 3.]], [[4., 5.],  [6., 7.]]]
           # example 1, [[[10., 11.],  [12., 13.]]]
           'indices': ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                       (0, 7), (1, 0), (1, 1), (1, 2), (1, 3)),
           'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
           'dense_shape': (2, 8)},
       'expected_dense_tensor': [
           [[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
           [[[10., 11.], [12., 13.]], [[0., 0.], [0., 0.]]]]},
      {'testcase_name': '3D',
       'sparse_input_args': {
           'indices': ((0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6),
                       (0, 1, 0), (0, 1, 2), (0, 1, 4), (0, 1, 6),
                       (1, 0, 0), (1, 0, 2), (1, 0, 4), (1, 0, 6)),
           'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
           'dense_shape': (2, 2, 8)},
       'expected_dense_tensor': [
           [[[0., 0.], [1., 0.]], [[2., 0.], [3., 0.]],
            [[4., 0.], [5., 0.]], [[6., 0.], [7., 0.]]],
           [[[10., 0.], [11., 0.]], [[12., 0.], [13., 0.]],
            [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]]]},
      )
  def test_get_dense_tensor_multi_dim(
      self, sparse_input_args, expected_dense_tensor):
    # [FUNCTION DOC]
    # Parameterized scenario (two- and three-dimensional input) for a column of per-step shape two
    # by two. Expects the flat values to be grouped into two by two blocks per timestep and padded
    # with zeros for shorter examples.
    """Tests get_sequence_dense_tensor for multi-dim numeric_column."""
    sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)
    numeric_column = sfc.sequence_numeric_column('aaa', shape=(2, 2))

    dense_tensor, _ = _get_sequence_dense_tensor(
        numeric_column, {'aaa': sparse_input})

    self.assertAllEqual(
        expected_dense_tensor, self.evaluate(dense_tensor))

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           # example 0, ids [2]
           # example 1, ids [0, 1]
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': (2., 0., 1.),
           'dense_shape': (2, 2)},
       'expected_sequence_length': [1, 2],
       'shape': (1,)},
      {'testcase_name': '3D',
       'inputs_args': {
           # example 0, ids [[2]]
           # example 1, ids [[0, 1], [2]]
           'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
           'values': (2., 0., 1., 2.),
           'dense_shape': (2, 2, 2)},
       'expected_sequence_length': [1, 2],
       'shape': (1,)},
      {'testcase_name': '2D_with_shape',
       'inputs_args': {
           # example 0, ids [2]
           # example 1, ids [0, 1]
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': (2., 0., 1.),
           'dense_shape': (2, 2)},
       'expected_sequence_length': [1, 1],
       'shape': (2,)},
      {'testcase_name': '3D_with_shape',
       'inputs_args': {
           # example 0, ids [[2]]
           # example 1, ids [[0, 1], [2]]
           'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
           'values': (2., 0., 1., 2.),
           'dense_shape': (2, 2, 2)},
       'expected_sequence_length': [1, 2],
       'shape': (2,)},
      )
  def test_sequence_length(self, inputs_args, expected_sequence_length, shape):
    # [FUNCTION DOC]
    # Parameterized scenario (two- and three-dimensional input, per-step shape one or two).
    # Expects the sequence lengths to account for grouping of values by the per-step element
    # count, with 64-bit integer dtype.
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    numeric_column = sfc.sequence_numeric_column('aaa', shape=shape)

    _, sequence_length = _get_sequence_dense_tensor(
        numeric_column, {'aaa': inputs})

    sequence_length = self.evaluate(sequence_length)
    self.assertAllEqual(expected_sequence_length, sequence_length)
    self.assertEqual(np.int64, sequence_length.dtype)

  def test_sequence_length_with_empty_rows(self):
    # [FUNCTION DOC]
    # Scenario: six examples where some have no values. Expects zero length for empty examples and
    # the value counts for the others.
    """Tests _sequence_length when some examples do not have ids."""
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, values []
        # example 1, values [[0.], [1.]]
        # example 2, [[2.]]
        # example 3, values []
        # example 4, [[3.]]
        # example 5, values []
        indices=((1, 0), (1, 1), (2, 0), (4, 0)),
        values=(0., 1., 2., 3.),
        dense_shape=(6, 2))
    expected_sequence_length = [0, 2, 1, 0, 1, 0]
    numeric_column = sfc.sequence_numeric_column('aaa')

    _, sequence_length = _get_sequence_dense_tensor(
        numeric_column, {'aaa': sparse_input})

    self.assertAllEqual(
        expected_sequence_length, self.evaluate(sequence_length))

  def test_serialization(self):
    # [FUNCTION DOC]
    # Scenario: a column with custom key, shape, default value, integer dtype and a custom
    # normalizer is serialized and deserialized with the normalizer supplied as a custom object.
    # Expects key, shape, default value and the normalizer behavior to survive; also expects a
    # value error when serializing something that is not a feature column.
    """Tests that column can be serialized."""
    def _custom_fn(input_tensor):
      # [FUNCTION DOC]
      # HELPER: custom normalizer for the serialization test. Takes a tensor and returns it plus
      # a constant of forty-two.
      return input_tensor + 42

    column = sfc.sequence_numeric_column(
        key='my-key', shape=(2,), default_value=3, dtype=dtypes.int32,
        normalizer_fn=_custom_fn)
    configs = serialization.serialize_feature_column(column)
    column = serialization.deserialize_feature_column(
        configs, custom_objects={_custom_fn.__name__: _custom_fn})
    self.assertEqual(column.key, 'my-key')
    self.assertEqual(column.shape, (2,))
    self.assertEqual(column.default_value, 3)
    self.assertEqual(column.normalizer_fn(3), 45)
    with self.assertRaisesRegex(ValueError,
                                'Instance: 0 is not a FeatureColumn'):
      serialization.serialize_feature_column(int())

  def test_parents(self):
    # [FUNCTION DOC]
    # Scenario: a column with a key. Expects its parents to be a list containing just that key.
    """Tests parents attribute of column."""
    column = sfc.sequence_numeric_column(key='my-key')
    self.assertEqual(column.parents, ['my-key'])


if __name__ == '__main__':
  test.main()
