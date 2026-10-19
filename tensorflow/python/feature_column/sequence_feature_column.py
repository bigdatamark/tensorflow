# [MODULE DOC]
# Sequence feature columns: factory functions for sequence categorical columns (identity, hash
# bucket, vocabulary file, vocabulary list) and sequence numeric columns, plus the sequence numeric
# column class and helpers. Categorical factories wrap the corresponding non-sequence column in a
# sequence categorical wrapper from the main feature column module. Public factories are exported
# in the feature-column API namespace and are marked deprecated in favor of Keras preprocessing
# layers. Import-time side effects: API export and deprecation decorators are applied and the
# numeric column class is registered with the feature column serialization registry.
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
"""This API defines FeatureColumn for sequential input.

NOTE: This API is a work in progress and will likely be changing frequently.
"""

import collections

from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import serialization
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls

_FEATURE_COLUMN_DEPRECATION_WARNING = """\
    Warning: tf.feature_column is not recommended for new code. Instead,
    feature preprocessing can be done directly using either [Keras preprocessing
    layers](https://www.tensorflow.org/guide/migrate/migrating_feature_columns)
    or through the one-stop utility [`tf.keras.utils.FeatureSpace`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/FeatureSpace)
    built on top of them. See the [migration guide](https://tensorflow.org/guide/migrate)
    for details.
    """

_FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING = (
    'Use Keras preprocessing layers instead, either directly or via the '
    '`tf.keras.utils.FeatureSpace` utility. Each of `tf.feature_column.*` has '
    'a functional equivalent in `tf.keras.layers` for feature preprocessing '
    'when training a Keras model.')


# pylint: disable=protected-access
def concatenate_context_input(context_input, sequence_input):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring lists only a value error, but wrong dtypes raise a type error; and
  # when ranks are not statically known the assertion ops fail at execution time instead.
  # Appends a per-example context vector to every timestep of a sequence tensor. Inputs: a
  # float32 context tensor of rank 2 (batch, features) and a float32 sequence tensor of rank 3
  # (batch, padded length, features). Output: a rank 3 float32 tensor whose last dimension is the
  # sum of both feature sizes. Graph assertions on rank and dtype are attached as control
  # dependencies; with statically known ranks a value error is raised at call time (type error for
  # dtype mismatches). Side effects: adds assertion, tile and concat operations to the current graph.
  """Replicates `context_input` across all timesteps of `sequence_input`.

  Expands dimension 1 of `context_input` then tiles it `sequence_length` times.
  This value is appended to `sequence_input` on dimension 2 and the result is
  returned.

  Args:
    context_input: A `Tensor` of dtype `float32` and shape `[batch_size, d1]`.
    sequence_input: A `Tensor` of dtype `float32` and shape `[batch_size,
      padded_length, d0]`.

  Returns:
    A `Tensor` of dtype `float32` and shape `[batch_size, padded_length,
    d0 + d1]`.

  Raises:
    ValueError: If `sequence_input` does not have rank 3 or `context_input` does
      not have rank 2.
  """
  seq_rank_check = check_ops.assert_rank(
      sequence_input,
      3,
      message='sequence_input must have rank 3',
      data=[array_ops.shape(sequence_input)])
  seq_type_check = check_ops.assert_type(
      sequence_input,
      dtypes.float32,
      message='sequence_input must have dtype float32; got {}.'.format(
          sequence_input.dtype))
  ctx_rank_check = check_ops.assert_rank(
      context_input,
      2,
      message='context_input must have rank 2',
      data=[array_ops.shape(context_input)])
  ctx_type_check = check_ops.assert_type(
      context_input,
      dtypes.float32,
      message='context_input must have dtype float32; got {}.'.format(
          context_input.dtype))
  with ops.control_dependencies(
      [seq_rank_check, seq_type_check, ctx_rank_check, ctx_type_check]):
    padded_length = array_ops.shape(sequence_input)[1]
    tiled_context_input = array_ops.tile(
        array_ops.expand_dims(context_input, 1),
        array_ops.concat([[1], [padded_length], [1]], 0))
  return array_ops.concat([sequence_input, tiled_context_input], 2)


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.sequence_categorical_column_with_identity')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def sequence_categorical_column_with_identity(key,
                                              num_buckets,
                                              default_value=None):
  # [FUNCTION DOC]
  # Public factory for a sequence categorical column whose ids are the raw integer inputs.
  # Inputs: key (unique feature name string), num_buckets (inputs must lie in zero up to this
  # value, exclusive), default_value (optional; replaces out-of-range ids, otherwise out-of-range
  # inputs fail at graph run time). Output: a sequence categorical column wrapping an identity
  # categorical column. Errors (invalid bucket count or default value) are raised by the wrapped
  # factory as value errors. Deprecated: calling it emits a deprecation warning.
  """Returns a feature column that represents sequences of integers.

  Pass this to `embedding_column` or `indicator_column` to convert sequence
  categorical data into dense representation for input to sequence NN, such as
  RNN.

  Example:

  ```python
  watches = sequence_categorical_column_with_identity(
      'watches', num_buckets=1000)
  watches_embedding = embedding_column(watches, dimension=10)
  columns = [watches_embedding]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
  ```

  Args:
    key: A unique string identifying the input feature.
    num_buckets: Range of inputs. Namely, inputs are expected to be in the range
      `[0, num_buckets)`.
    default_value: If `None`, this column's graph operations will fail for
      out-of-range inputs. Otherwise, this value must be in the range `[0,
      num_buckets)`, and will replace out-of-range inputs.

  Returns:
    A `SequenceCategoricalColumn`.

  Raises:
    ValueError: if `num_buckets` is less than one.
    ValueError: if `default_value` is not in range `[0, num_buckets)`.
  """
  return fc.SequenceCategoricalColumn(
      fc.categorical_column_with_identity(
          key=key, num_buckets=num_buckets, default_value=default_value))


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.sequence_categorical_column_with_hash_bucket')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def sequence_categorical_column_with_hash_bucket(key,
                                                 hash_bucket_size,
                                                 dtype=dtypes.string):
  # [FUNCTION DOC]
  # Public factory for a sequence categorical column whose ids come from hashing the input
  # values. Inputs: key (unique feature name string), hash_bucket_size (number of buckets,
  # must exceed one), dtype (string by default; string or integer types only). Output: a
  # sequence categorical column wrapping a hash-bucket categorical column. Value errors for a bad
  # bucket size or dtype are raised by the wrapped factory. Deprecated: emits a deprecation warning.
  """A sequence of categorical terms where ids are set by hashing.

  Pass this to `embedding_column` or `indicator_column` to convert sequence
  categorical data into dense representation for input to sequence NN, such as
  RNN.

  Example:

  ```python
  tokens = sequence_categorical_column_with_hash_bucket(
      'tokens', hash_bucket_size=1000)
  tokens_embedding = embedding_column(tokens, dimension=10)
  columns = [tokens_embedding]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
  ```

  Args:
    key: A unique string identifying the input feature.
    hash_bucket_size: An int > 1. The number of buckets.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `SequenceCategoricalColumn`.

  Raises:
    ValueError: `hash_bucket_size` is not greater than 1.
    ValueError: `dtype` is neither string nor integer.
  """
  return fc.SequenceCategoricalColumn(
      fc.categorical_column_with_hash_bucket(
          key=key, hash_bucket_size=hash_bucket_size, dtype=dtype))


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.sequence_categorical_column_with_vocabulary_file')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def sequence_categorical_column_with_vocabulary_file(key,
                                                     vocabulary_file,
                                                     vocabulary_size=None,
                                                     num_oov_buckets=0,
                                                     default_value=None,
                                                     dtype=dtypes.string):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says default_value defaults to minus one, but the signature
  # default is nothing/null; the wrapped factory interprets null as its own default.
  # Public factory for a sequence categorical column whose ids come from a vocabulary file.
  # Inputs: key (feature name), vocabulary_file (path), vocabulary_size (optional; number of
  # entries to use, defaults to whole file), num_oov_buckets (non-negative count of hashed
  # out-of-vocabulary buckets, default zero), default_value (optional id for unknown values; cannot
  # be combined with positive oov buckets), dtype (string by default; string or integer).
  # Output: a sequence categorical column wrapping a vocabulary-file categorical column. Value
  # errors for bad arguments are raised by the wrapped factory. Deprecated: emits a warning.
  """A sequence of categorical terms where ids use a vocabulary file.

  Pass this to `embedding_column` or `indicator_column` to convert sequence
  categorical data into dense representation for input to sequence NN, such as
  RNN.

  Example:

  ```python
  states = sequence_categorical_column_with_vocabulary_file(
      key='states', vocabulary_file='/us/states.txt', vocabulary_size=50,
      num_oov_buckets=5)
  states_embedding = embedding_column(states, dimension=10)
  columns = [states_embedding]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
  ```

  Args:
    key: A unique string identifying the input feature.
    vocabulary_file: The vocabulary file name.
    vocabulary_size: Number of the elements in the vocabulary. This must be no
      greater than length of `vocabulary_file`, if less than length, later
      values are ignored. If None, it is set to the length of `vocabulary_file`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of
      the input value. A positive `num_oov_buckets` can not be specified with
      `default_value`.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `SequenceCategoricalColumn`.

  Raises:
    ValueError: `vocabulary_file` is missing or cannot be opened.
    ValueError: `vocabulary_size` is missing or < 1.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: `dtype` is neither string nor integer.
  """
  return fc.SequenceCategoricalColumn(
      fc.categorical_column_with_vocabulary_file(
          key=key,
          vocabulary_file=vocabulary_file,
          vocabulary_size=vocabulary_size,
          num_oov_buckets=num_oov_buckets,
          default_value=default_value,
          dtype=dtype))


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.sequence_categorical_column_with_vocabulary_list')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def sequence_categorical_column_with_vocabulary_list(key,
                                                     vocabulary_list,
                                                     dtype=None,
                                                     default_value=-1,
                                                     num_oov_buckets=0):
  # [FUNCTION DOC]
  # Public factory for a sequence categorical column whose ids are positions in an in-memory
  # vocabulary list. Inputs: key (feature name), vocabulary_list (ordered, duplicate-free
  # iterable), dtype (optional; inferred from the list when null), default_value (id for unknown
  # values, default minus one; cannot be combined with positive oov buckets), num_oov_buckets
  # (hashed out-of-vocabulary buckets, default zero). Output: a sequence categorical column
  # wrapping a vocabulary-list categorical column. Value errors for bad arguments come from the
  # wrapped factory. Deprecated: emits a deprecation warning.
  """A sequence of categorical terms where ids use an in-memory list.

  Pass this to `embedding_column` or `indicator_column` to convert sequence
  categorical data into dense representation for input to sequence NN, such as
  RNN.

  Example:

  ```python
  colors = sequence_categorical_column_with_vocabulary_list(
      key='colors', vocabulary_list=('R', 'G', 'B', 'Y'),
      num_oov_buckets=2)
  colors_embedding = embedding_column(colors, dimension=3)
  columns = [colors_embedding]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
  ```

  Args:
    key: A unique string identifying the input feature.
    vocabulary_list: An ordered iterable defining the vocabulary. Each feature
      is mapped to the index of its value (if present) in `vocabulary_list`.
      Must be castable to `dtype`.
    dtype: The type of features. Only string and integer types are supported. If
      `None`, it will be inferred from `vocabulary_list`.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[len(vocabulary_list), len(vocabulary_list)+num_oov_buckets)` based on a
      hash of the input value. A positive `num_oov_buckets` can not be specified
      with `default_value`.

  Returns:
    A `SequenceCategoricalColumn`.

  Raises:
    ValueError: if `vocabulary_list` is empty, or contains duplicate keys.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: if `dtype` is not integer or string.
  """
  return fc.SequenceCategoricalColumn(
      fc.categorical_column_with_vocabulary_list(
          key=key,
          vocabulary_list=vocabulary_list,
          dtype=dtype,
          default_value=default_value,
          num_oov_buckets=num_oov_buckets))


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.sequence_numeric_column')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def sequence_numeric_column(key,
                            shape=(1,),
                            default_value=0.,
                            dtype=dtypes.float32,
                            normalizer_fn=None):
  # [FUNCTION DOC]
  # Public factory for a sequence numeric column. Inputs: key (feature name), shape (per-step
  # value shape, default a single value), default_value (padding value when densifying, default
  # zero), dtype (float32 by default; must be integer or floating), normalizer_fn (optional
  # callable applied to the tensor). Output: a sequence numeric column. Raises a value error for an
  # invalid shape or non-numeric dtype and a type error for a non-callable normalizer or non-integer
  # shape dimension. Deprecated: emits a deprecation warning.
  # KNOWN ISSUE: the docstring says dtype must be convertible to float32; the code only checks the
  # dtype is integer or floating and does not convert.
  """Returns a feature column that represents sequences of numeric data.

  Example:

  ```python
  temperature = sequence_numeric_column('temperature')
  columns = [temperature]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
  ```

  Args:
    key: A unique string identifying the input features.
    shape: The shape of the input data per sequence id. E.g. if `shape=(2,)`,
      each example must contain `2 * sequence_length` values.
    default_value: A single value compatible with `dtype` that is used for
      padding the sparse data into a dense `Tensor`.
    dtype: The type of values.
    normalizer_fn: If not `None`, a function that can be used to normalize the
      value of the tensor after `default_value` is applied for parsing.
      Normalizer function takes the input `Tensor` as its argument, and returns
      the output `Tensor`. (e.g. lambda x: (x - 3.0) / 4.2). Please note that
      even though the most common use case of this function is normalization, it
      can be used for any kind of Tensorflow transformations.

  Returns:
    A `SequenceNumericColumn`.

  Raises:
    TypeError: if any dimension in shape is not an int.
    ValueError: if any dimension in shape is not a positive integer.
    ValueError: if `dtype` is not convertible to `tf.float32`.
  """
  shape = fc._check_shape(shape=shape, key=key)
  if not (dtype.is_integer or dtype.is_floating):
    raise ValueError('dtype must be convertible to float. '
                     'dtype: {}, key: {}'.format(dtype, key))
  if normalizer_fn is not None and not callable(normalizer_fn):
    raise TypeError(
        'normalizer_fn must be a callable. Given: {}'.format(normalizer_fn))

  return SequenceNumericColumn(
      key,
      shape=shape,
      default_value=default_value,
      dtype=dtype,
      normalizer_fn=normalizer_fn)


def _assert_all_equal_and_return(tensors, name=None):
  # [FUNCTION DOC]
  # HELPER: internal utility for sequence feature column code (not referenced elsewhere in this file).
  # Takes a list of tensors and an optional name scope. If there is one tensor it is returned
  # unchanged; otherwise equality assertions between the first tensor and each other are added
  # and an identity of the first tensor is returned under those control dependencies, so
  # mismatches fail at run time. Side effects: adds operations to the current graph.
  """Asserts that all tensors are equal and returns the first one."""
  with ops.name_scope(name, 'assert_all_equal', values=tensors):
    if len(tensors) == 1:
      return tensors[0]
    assert_equal_ops = []
    for t in tensors[1:]:
      assert_equal_ops.append(check_ops.assert_equal(tensors[0], t))
    with ops.control_dependencies(assert_equal_ops):
      return array_ops.identity(tensors[0])


@serialization.register_feature_column
class SequenceNumericColumn(
    fc.SequenceDenseColumn,
    collections.namedtuple(
        'SequenceNumericColumn',
        ('key', 'shape', 'default_value', 'dtype', 'normalizer_fn'))):
  # [CLASS DOC]
  # Sequence dense column for numeric data, built from a named tuple holding key, shape,
  # default value, dtype and normalizer function. Parses variable-length sparse values for its
  # key, optionally normalizes them, and densifies into a padded tensor with per-example sequence
  # lengths. Created by the sequence numeric factory or from a config. Registered with the feature
  # column serialization registry.
  """Represents sequences of numeric data."""

  @property
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # HELPER: internal compatibility flag. Always reports true: this column follows the new
    # column API. No inputs, no side effects.
    return True

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property returning the column name, which is simply the feature key string.
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # Property returning a mapping with a single entry: the feature key mapped to a
    # variable-length feature of the column's dtype, so values parse into a sparse tensor.
    """See `FeatureColumn` base class."""
    return {self.key: parsing_ops.VarLenFeature(self.dtype)}

  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Fetches the raw sparse tensor for the column key from the transformation cache and, when a
    # normalizer function is set, applies it. Inputs: the transformation cache and a state manager
    # (unused here). Output: the possibly normalized tensor. No other side effects.
    """See `FeatureColumn` base class.

    In this case, we apply the `normalizer_fn` to the input tensor.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Normalized input tensor.
    """
    input_tensor = transformation_cache.get(self.key, state_manager)
    if self.normalizer_fn is not None:
      input_tensor = self.normalizer_fn(input_tensor)
    return input_tensor

  @property
  def variable_shape(self):
    # [FUNCTION DOC]
    # Property returning the per-timestep value shape as a tensor shape object built from the
    # column's shape setting.
    """Returns a `TensorShape` representing the shape of sequence input."""
    return tensor_shape.TensorShape(self.shape)

  def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Builds the dense padded sequence and its lengths. Gets the column's transformed sparse
    # tensor from the cache, converts it to dense using the default value as padding, and reshapes
    # it to batch, time, and the per-step shape. Sequence length is computed from the sparse
    # tensor, dividing by the number of elements per step when the sparse input is rank two (rank
    # three input counts steps directly). Inputs: transformation cache, state manager (unused).
    # Output: a pair holding the dense tensor and the per-example length tensor. Adds graph ops.
    # KNOWN ISSUE: the docstring lacks a Returns section and does not describe the length output.
    """Returns a `TensorSequenceLengthPair`.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.
    """
    sp_tensor = transformation_cache.get(self, state_manager)
    dense_tensor = sparse_ops.sparse_tensor_to_dense(
        sp_tensor, default_value=self.default_value)
    # Reshape into [batch_size, T, variable_shape].
    dense_shape = array_ops.concat(
        [array_ops.shape(dense_tensor)[:1], [-1], self.variable_shape], axis=0)
    dense_tensor = array_ops.reshape(dense_tensor, shape=dense_shape)

    # Get the number of timesteps per example
    # For the 2D case, the raw values are grouped according to num_elements;
    # for the 3D case, the grouping happens in the third dimension, and
    # sequence length is not affected.
    if sp_tensor.shape.ndims == 2:
      num_elements = self.variable_shape.num_elements()
    else:
      num_elements = 1
    seq_length = fc_utils.sequence_length_from_sparse_tensor(
        sp_tensor, num_elements=num_elements)

    return fc.SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=seq_length)

  @property
  def parents(self):
    # [FUNCTION DOC]
    # Property returning a one-element list with the raw feature key; this column depends on no
    # other columns.
    """See 'FeatureColumn` base class."""
    return [self.key]

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping of the column's fields to their values, with the dtype replaced by its
    # name string. KNOWN ISSUE: the normalizer function is left in the mapping as is and is not
    # serialized, so the result is not fully serializable when a normalizer is set.
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # Class method rebuilding the column from a config mapping. Validates that the config keys
    # match the column's fields, makes a standardized copy of the config, converts the dtype name
    # back to a dtype and constructs the column. The custom objects and columns-by-name arguments are
    # accepted for interface compatibility but ignored. Raises an error if config keys are invalid.
    # KNOWN ISSUE: a normalizer function in the config is passed through without deserialization.
    """See 'FeatureColumn` base class."""
    fc._check_config_keys(config, cls._fields)
    kwargs = fc._standardize_and_copy_config(config)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)


# pylint: enable=protected-access
