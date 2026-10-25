# [MODULE DOC]
# TPU feature column library. Provides factory functions that build TPU-compatible embedding
# columns and shared embedding columns from categorical feature columns, plus the internal
# column classes (base mixin, embedding, shared embedding) that wrap the core feature column
# types so that, when building for TPU, lookups are left to the TPU embedding engine and the
# variable names and scopes are recorded in a graph collection. Also holds the supported and
# denylisted categorical column type lists and a helper to name sequence length features.
# Import time: only defines module constants (no registrations).
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
# ===================================================================
"""TPU Feature Column Library."""
import math

from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_replication
# pylint: disable=protected-access


_TPU_FC_TO_SCOPE = '_tpu_feature_column_scope'
_SUPPORTED_SEQUENCE_COLUMNS = (fc._SequenceCategoricalColumn,
                               fc_lib.SequenceCategoricalColumn)


# For V2 columns, we support anything that inherits from CategoricalColumn
# other than those in the denylist. User-provided columns that inherit from
# CategoricalColumn may or may not be compatible; it is up to the user to
# manage TPU compatibility for custom columns.
_SUPPORTED_CATEGORICAL_COLUMNS_V2 = (fc_lib.CategoricalColumn,)
_DENYLISTED_CATEGORICAL_COLUMNS_V2 = (fc_lib.HashedCategoricalColumn,
                                      fc_lib.BucketizedColumn,
                                      fc_lib.CrossedColumn)
_SUPPORTED_CATEGORICAL_COLUMNS = (fc._IdentityCategoricalColumn,
                                  fc._VocabularyFileCategoricalColumn,
                                  fc._VocabularyListCategoricalColumn,
                                  fc._WeightedCategoricalColumn,
                                  fc._SequenceCategoricalColumn
                                 ) + _SUPPORTED_CATEGORICAL_COLUMNS_V2
_SEQUENCE_FEATURE_LENGTH_POSTFIX = '_seq_length_'


def embedding_column(categorical_column,
                     dimension,
                     combiner='mean',
                     initializer=None,
                     max_sequence_length=0,
                     learning_rate_fn=None,
                     use_safe_embedding_lookup=True):
  # [FUNCTION DOC]
  # Factory creating a TPU embedding column from a categorical column. Inputs:
  # categorical_column, a supported categorical column; dimension, a positive integer;
  # combiner (default mean); initializer, optional callable (default truncated normal with
  # standard deviation one over the square root of the dimension); max_sequence_length,
  # non-negative integer (positive for sequence columns, zero otherwise); learning_rate_fn,
  # optional; use_safe_embedding_lookup, flag (default true).
  # Builds a layer-creator closure and the TPU embedding column, then attaches the initializer
  # to a private attribute because the creator hides it. Returns the column.
  # Raises a type error for denylisted or unsupported categorical column types, and value
  # errors for a missing or non-positive dimension, a non-callable initializer, or inconsistent
  # sequence length.
  """TPU embedding_column for `tf.feature_column.embedding_column`.

  Note that the interface for TPU embedding_column is different from the non-TPU
  version. The following args available for the non-TPU version are NOT
  supported: ckpt_to_load_from, tensor_name_in_ckp, max_norm and trainable.

  Args:
    categorical_column: A categorical_column returned from
        categorical_column_with_identity, weighted_categorical_column,
        categorical_column_with_vocabulary_file,
        categorical_column_with_vocabulary_list,
        sequence_categorical_column_with_identity,
        sequence_categorical_column_with_vocabulary_file,
        sequence_categorical_column_with_vocabulary_list
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row for a non-sequence column. For more information, see
      `tf.feature_column.embedding_column`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.compat.v1.truncated_normal_initializer` with mean `0.0` and
      standard deviation `1/sqrt(dimension)`.
    max_sequence_length: An non-negative integer specifying the max sequence
      length. Any sequence shorter then this will be padded with 0 embeddings
      and any sequence longer will be truncated. This must be positive for
      sequence features and 0 for non-sequence features.
    learning_rate_fn: A function that takes global step and returns learning
      rate for the embedding table. If you intend to use the same learning rate
      for multiple embedding tables, please ensure that you pass the exact same
      python function to all calls of embedding_column, otherwise performence
      may suffer.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies to rank 2 (NxM) shaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.

  Returns:
    A  _TPUEmbeddingColumn.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if `initializer` is specified but not callable.
    TypeError: if categorical_column is not a supported type.
  """
  if isinstance(categorical_column, _DENYLISTED_CATEGORICAL_COLUMNS_V2):
    raise TypeError('categorical_column for tpu '
                    ' embedding_column was '
                    f'denylisted type {type(categorical_column)}')
  if not isinstance(categorical_column, _SUPPORTED_CATEGORICAL_COLUMNS):
    raise TypeError(
        'categorical_column for tpu '
        ' embedding_column must be type {}, got {}.'.format(' or '.join([
            cc.__name__ for cc in _SUPPORTED_CATEGORICAL_COLUMNS
        ]), type(categorical_column)))
  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified. '
                     'Embedding of column_name: {}'.format(
                         categorical_column.name))
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1 / math.sqrt(dimension))

  embedding_shape = categorical_column._num_buckets, dimension  # pylint: disable=protected-access

  def _creator(weight_collections, scope):
    # [FUNCTION DOC]
    # HELPER: layer creator handed to the embedding column. Inputs: weight_collections, the
    # collections for the new weights, and scope, the variable scope. Builds an embedding
    # column layer for the captured table shape and initializer (trainable, fixed layer name),
    # calls it with no input in the given scope, and returns the result. Side effect: creates
    # the embedding variable.
    embedding_column_layer = fc._EmbeddingColumnLayer(
        embedding_shape=embedding_shape,
        initializer=initializer,
        weight_collections=weight_collections,
        trainable=True,
        name='embedding_column_layer')
    return embedding_column_layer(None, scope=scope)  # pylint: disable=not-callable

  column = _TPUEmbeddingColumn(
      categorical_column=categorical_column,
      dimension=dimension,
      combiner=combiner,
      layer_creator=_creator,
      ckpt_to_load_from=None,
      tensor_name_in_ckpt=None,
      max_norm=None,
      trainable=True,
      max_sequence_length=max_sequence_length,
      learning_rate_fn=learning_rate_fn,
      use_safe_embedding_lookup=use_safe_embedding_lookup)
  # For Embedding column, the initializer is hidden inside the creator Fn, which
  # is not accessible later. So, we attach it to a special field. Also note
  # that non-TPU Embedding column and non-TPU shared Embedding column handle the
  # initializer differently. See shared_embedding_columns for details.
  column._tpu_initializer = initializer
  return column


def shared_embedding_columns(categorical_columns,
                             dimension,
                             combiner='mean',
                             initializer=None,
                             shared_embedding_collection_name=None,
                             max_sequence_lengths=None,
                             learning_rate_fn=None,
                             use_safe_embedding_lookup=True):
  # [FUNCTION DOC]
  # Factory creating one TPU shared embedding column per categorical column, all sharing a
  # single embedding table. Inputs: categorical_columns, a list of supported categorical
  # columns with identical bucket counts; dimension, a positive integer; combiner (default
  # mean); initializer, optional callable (default truncated normal with standard deviation
  # one over the square root of the dimension); shared_embedding_collection_name, optional
  # name (default: sorted column names joined plus a shared embedding suffix);
  # max_sequence_lengths, optional list matching the columns (null or empty means all zero);
  # learning_rate_fn, optional; use_safe_embedding_lookup, flag (default true).
  # Returns a list of shared TPU embedding columns in input order.
  # Raises a type error for denylisted or unsupported column types, and value errors for a
  # length mismatch, invalid dimension, non-callable initializer, differing bucket counts, or
  # sequence length inconsistencies detected by the column constructor.
  # KNOWN ISSUE: the docstring says it returns a single TPU embedding column type, but the
  # function returns a list of shared embedding columns; its Raises section also omits the type
  # errors raised for unsupported column types.
  """List of dense columns that convert from sparse, categorical input.

  Note that the interface for TPU embedding_column is different from the non-TPU
  version. The following args available for the non-TPU version are NOT
  supported: ckpt_to_load_from, tensor_name_in_ckp, max_norm and trainable.

  Args:
    categorical_columns: A list of categorical_columns returned from
        categorical_column_with_identity, weighted_categorical_column,
        categorical_column_with_vocabulary_file,
        categorical_column_with_vocabulary_list,
        sequence_categorical_column_with_identity,
        sequence_categorical_column_with_vocabulary_file,
        sequence_categorical_column_with_vocabulary_list
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row for a non-sequence column. For more information, see
      `tf.feature_column.embedding_column`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional name of the collection where
      shared embedding weights are added. If not given, a reasonable name will
      be chosen based on the names of `categorical_columns`. This is also used
      in `variable_scope` when creating shared embedding weights.
    max_sequence_lengths: An list of non-negative integers, either None or
      empty or the same length as the argument categorical_columns. Entries
      corresponding to non-sequence columns must be 0 and entries corresponding
      to sequence columns specify the max sequence length for the column. Any
      sequence shorter then this will be padded with 0 embeddings and any
      sequence longer will be truncated.
    learning_rate_fn: A function that takes global step and returns learning
      rate for the embedding table. If you intend to use the same learning rate
      for multiple embedding tables, please ensure that you pass the exact same
      python function to all calls of shared_embedding_columns, otherwise
      performence may suffer.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies to rank 2 (NxM) shaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.

  Returns:
    A  _TPUEmbeddingColumn.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if `initializer` is specified but not callable.
    ValueError: if `max_sequence_lengths` is specified and not the same length
      as `categorical_columns`.
    ValueError: if `max_sequence_lengths` is positive for a non sequence column
      or 0 for a sequence column.
  """
  for categorical_column in categorical_columns:
    if isinstance(categorical_column, _DENYLISTED_CATEGORICAL_COLUMNS_V2):
      raise TypeError('categorical_column for tpu '
                      ' embedding_column was denylisted type '
                      f'{type(categorical_column)}')
    if not isinstance(categorical_column, _SUPPORTED_CATEGORICAL_COLUMNS):
      raise TypeError(
          'categorical_column for tpu '
          ' shared_embedding_columns must be type {}, got {}.'.format(
              ' or '.join(
                  [cc.__name__ for cc in _SUPPORTED_CATEGORICAL_COLUMNS]),
              type(categorical_column)))

  if not max_sequence_lengths:
    max_sequence_lengths = [0] * len(categorical_columns)
  if len(max_sequence_lengths) != len(categorical_columns):
    raise ValueError('max_sequence_lengths and categorical_columns must be of '
                     'the same length. len(max_sequence_lengths)={} '
                     'len(categorical_columns)={}.'.format(
                         len(max_sequence_lengths), len(categorical_columns)))

  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified. ')
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1 / math.sqrt(dimension))

  # Sort the columns so the default collection name is deterministic even if the
  # user passes columns from an unsorted collection, such as dict.values().
  sorted_columns = sorted(categorical_columns, key=lambda x: x.name)
  num_buckets = sorted_columns[0]._num_buckets  # pylint: disable=protected-access

  for c in sorted_columns[1:]:
    if num_buckets != c._num_buckets:  # pylint: disable=protected-access
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same number of buckets. Given column: {} with buckets: {} does  '
          'not match column: {} with buckets: {}'.format(
              sorted_columns[0], num_buckets, c, c._num_buckets))  # pylint: disable=protected-access

  if not shared_embedding_collection_name:
    shared_embedding_collection_name = '_'.join(c.name for c in sorted_columns)
    shared_embedding_collection_name += '_shared_embedding'

  tpu_columns = []

  # Create the state (_SharedEmbeddingColumnLayer) here.
  for categorical_column, max_sequence_length in zip(
      categorical_columns, max_sequence_lengths):
    column = _TPUSharedEmbeddingColumn(
        categorical_column=categorical_column,
        dimension=dimension,
        combiner=combiner,
        initializer=initializer,
        shared_embedding_collection_name=shared_embedding_collection_name,
        ckpt_to_load_from=None,
        tensor_name_in_ckpt=None,
        max_norm=None,
        trainable=True,
        max_sequence_length=max_sequence_length,
        learning_rate_fn=learning_rate_fn,
        use_safe_embedding_lookup=use_safe_embedding_lookup)
    tpu_columns.append(column)

  return tpu_columns


class _TPUBaseEmbeddingColumn(object):
  # [CLASS DOC]
  # HELPER: mixin base for the TPU embedding column types.
  # Stores the wrapped categorical column, the maximum sequence length and the optional
  # learning rate function, validates sequence length consistency, and declares abstract
  # accessors (combiner, table size, feature key, weight key, variable name, initializer,
  # weighted check) that concrete subclasses implement. Used by the TPU embedding configuration
  # code to read column properties.
  """Base class for TPU Embedding Column."""

  def __init__(self,
               categorical_column,
               max_sequence_length=0,
               learning_rate_fn=None):
    # [FUNCTION DOC]
    # Initializer. Inputs: categorical_column, the wrapped categorical column; max_sequence_length,
    # a non-negative integer (default zero, required positive for sequence columns and zero
    # otherwise); learning_rate_fn, optional function from global step to learning rate.
    # Stores all three. Raises a value error if a sequence column has a length below one or a
    # non-sequence column has a non-zero length.
    self._tpu_categorical_column = categorical_column
    self._max_sequence_length = max_sequence_length
    self._learning_rate_fn = learning_rate_fn
    if (self.is_sequence_column() and max_sequence_length < 1):
      raise ValueError('max_sequence_length must be greater than 0 for '
                       'sequence columns. Got max_sequence_length={} for '
                       'sequence column {}.'.format(max_sequence_length,
                                                    categorical_column.name))
    if (not self.is_sequence_column() and max_sequence_length != 0):
      raise ValueError('Non zero max_seq_length={} specified for non '
                       'sequence column {}.'.format(max_sequence_length,
                                                    categorical_column.name))

  def get_combiner(self):
    # [FUNCTION DOC]
    # Abstract accessor for the combiner name; subclasses override it. This version always
    # raises a not-implemented error.
    """Returns the embedding combiner."""
    raise NotImplementedError('not implemented')

  def get_embedding_table_size(self):
    # [FUNCTION DOC]
    # Abstract accessor for the table size (vocabulary size and dimension); subclasses override
    # it. This version always raises a not-implemented error.
    """Returns the embedding table size, tuple of vocab size and dimension."""
    raise NotImplementedError('not implemented')

  def get_feature_key_name(self):
    # [FUNCTION DOC]
    # Abstract accessor for the input feature name; subclasses override it. This version always
    # raises a not-implemented error.
    """Returns the feature key name in the features dict."""
    raise NotImplementedError('not impl')

  def get_weight_key_name(self):
    # [FUNCTION DOC]
    # Abstract accessor for the weight feature name; subclasses override it. This version
    # always raises a not-implemented error.
    """Return the key name for weights."""
    raise NotImplementedError('not impl')

  def get_embedding_var_name(self):
    # [FUNCTION DOC]
    # Abstract accessor for the embedding table variable name; subclasses override it. This
    # version always raises a not-implemented error.
    """Returns the embedding variable name.

    Feature key name and embedding variable name are usually one-to-one mapping.
    But for shared embedding columns, it is many-to-one mapping.
    """
    raise NotImplementedError('not impl')

  def get_initializer(self):
    # [FUNCTION DOC]
    # Abstract accessor for the initializer; subclasses override it. This version always raises
    # a not-implemented error.
    """Returns the initializer."""
    raise NotImplementedError('not impl')

  def is_categorical_column_weighted(self):
    # [FUNCTION DOC]
    # Abstract predicate for whether the categorical column is weighted; subclasses override
    # it. This version always raises a not-implemented error.
    """Check if the categorical column of the embedding column is weighted."""
    raise NotImplementedError('not impl')

  def is_sequence_column(self):
    # [FUNCTION DOC]
    # Returns true if the wrapped categorical column is one of the supported sequence
    # categorical column types, otherwise false.
    return isinstance(self._tpu_categorical_column, _SUPPORTED_SEQUENCE_COLUMNS)

  def get_max_sequence_length(self):
    # [FUNCTION DOC]
    # Returns the maximum sequence length given at construction (zero for non-sequence columns).
    return self._max_sequence_length

  def get_learning_rate_fn(self):
    # [FUNCTION DOC]
    # Returns the learning rate function given at construction, or the null value if none.
    return self._learning_rate_fn

  def get_sequence_length_feature_key_name(self):
    # [FUNCTION DOC]
    # Returns the feature key of the companion sequence length feature, derived from this
    # column's feature key name by appending the sequence length suffix.
    """Get the key for the associated sequence length feature."""
    return get_sequence_length_feature_key_name_from_feature_key_name(
        self.get_feature_key_name())


class _TPUEmbeddingColumn(_TPUBaseEmbeddingColumn, fc._EmbeddingColumn):
  # [CLASS DOC]
  # HELPER: internal column type created by the embedding column factory.
  # TPU variant of the core embedding column combining the TPU base mixin with the parent
  # embedding column. Holds the sequence length, learning rate function, a scope validation
  # bypass flag and (attached later by the factory) the initializer. Overrides the dense and
  # sequence tensor methods so that, in TPU mode, the lookup is done by the TPU embedding
  # engine and only the variable name and scope are recorded.
  """Core Embedding Column."""

  def __new__(cls,
              categorical_column,
              dimension,
              combiner='mean',
              layer_creator=None,
              ckpt_to_load_from=None,
              tensor_name_in_ckpt=None,
              max_norm=None,
              trainable=True,
              max_sequence_length=0,
              learning_rate_fn=None,
              use_safe_embedding_lookup=True,
              bypass_scope_validation=False):
    # [FUNCTION DOC]
    # Allocation method. Forwards the embedding configuration (categorical column, dimension,
    # combiner, layer creator, unsupported checkpoint and norm options, trainable and safe
    # lookup flags) to the parent embedding column allocator, using this class as the type.
    # The bypass scope validation flag is deleted here (handled by the initializer), and the
    # sequence length and learning rate arguments are not forwarded. Returns the new instance.
    # Note, args ckpt_to_load_from, tensor_name_in_ckpt, max_norm and trainable
    # are not supported on TPU. They are solely for matching the signature of
    # __new__ of parent class fc._EmbeddingColumn.
    del bypass_scope_validation
    # pylint: disable=redundant-keyword-arg
    return fc._EmbeddingColumn.__new__(
        cls,
        categorical_column,
        dimension,
        combiner=combiner,
        layer_creator=layer_creator,
        ckpt_to_load_from=ckpt_to_load_from,
        tensor_name_in_ckpt=tensor_name_in_ckpt,
        max_norm=max_norm,
        trainable=trainable,
        use_safe_embedding_lookup=use_safe_embedding_lookup)

  def __init__(self,
               categorical_column,
               dimension,
               combiner='mean',
               layer_creator=None,
               ckpt_to_load_from=None,
               tensor_name_in_ckpt=None,
               max_norm=None,
               trainable=True,
               max_sequence_length=0,
               learning_rate_fn=None,
               use_safe_embedding_lookup=True,
               bypass_scope_validation=False):
    # [FUNCTION DOC]
    # Initializer. Passes categorical_column, max_sequence_length and learning_rate_fn to the
    # base initializer (which validates the sequence length against the column kind and may
    # raise a value error), resets the key to null and stores the scope validation bypass flag
    # (when true the same column may be used under several variable scopes). Other arguments
    # were consumed by the allocation method and are ignored here.
    _TPUBaseEmbeddingColumn.__init__(
        self,
        categorical_column,
        max_sequence_length=max_sequence_length,
        learning_rate_fn=learning_rate_fn)
    self._key = None
    # If true, scope validation is skipped to allow the same column to be used
    # in multiple variable scopes. By default, this is False, and we expect a
    # 1:1 mapping between feature columns and scopes.
    self._bypass_scope_validation = bypass_scope_validation

  def get_combiner(self):
    # [FUNCTION DOC]
    # Returns the combiner name (how multiple ids in a row are reduced) stored on the column.
    return self.combiner

  def get_embedding_table_size(self):
    # [FUNCTION DOC]
    # Returns a pair of the number of buckets of the wrapped categorical column (vocabulary
    # size) and the embedding dimension.
    """Returns num_ids and width."""
    return (self.categorical_column._num_buckets, self.dimension)

  def get_feature_key_name(self):
    # [FUNCTION DOC]
    # Returns the name of the input feature that feeds this column: the inner categorical
    # column's name when the column is weighted, else the categorical column's own name.
    """get_feature_key_name."""
    if self.is_categorical_column_weighted():
      return self.categorical_column.categorical_column.name
    return self.categorical_column.name

  def get_weight_key_name(self):
    # [FUNCTION DOC]
    # Returns the feature key holding the per-id weights if the categorical column is weighted,
    # otherwise returns the null value.
    """get_weight_key_name."""
    if self.is_categorical_column_weighted():
      return self.categorical_column.weight_feature_key
    return None

  def get_embedding_var_name(self):
    # [FUNCTION DOC]
    # Returns the name of the embedding table variable, which is the wrapped categorical
    # column's name (one table per column).
    """get_embedding_var_name."""
    return self.categorical_column.name

  def get_initializer(self):
    # [FUNCTION DOC]
    # Returns the initializer stored in the private TPU initializer attribute. That attribute
    # is attached by the embedding column factory function after construction.
    # KNOWN ISSUE: the attribute is not set by the constructor, so calling this on a column
    # created directly (not via the factory) raises an attribute error.
    return self._tpu_initializer

  def is_categorical_column_weighted(self):
    # [FUNCTION DOC]
    # Returns true if the wrapped categorical column is a weighted categorical column (old or
    # new style), otherwise false.
    """Check if the categorical column of the embedding column is weighted."""
    if isinstance(
        self.categorical_column,
        (
            fc._WeightedCategoricalColumn,  # pylint: disable=protected-access
            fc_lib.WeightedCategoricalColumn)):
      return True
    return False

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    # [FUNCTION DOC]
    # Produces the dense embedding tensor for this column. Inputs: inputs, the lazy feature
    # builder; weight_collections and trainable, passed to the parent on CPU paths. Under TPU
    # inference the parent computation runs on the host through outside compilation; on CPU
    # the parent implementation is used directly. In TPU training mode it returns the
    # already looked-up tensor fetched from inputs by feature key and records the embedding
    # variable name and variable scope in a graph collection (honoring the scope validation
    # bypass flag). Side effect: mutates that graph collection; may raise a value error on a
    # scope mismatch.
    if tpu.under_tpu_inference_context():
      def host_computation():
        # [FUNCTION DOC]
        # HELPER: host-side closure that calls the parent embedding column dense tensor logic
        # with the captured arguments and returns its tensor.
        return fc._EmbeddingColumn._get_dense_tensor(
            self, inputs, weight_collections, trainable)

      return tpu_replication.outside_compilation(host_computation)

    if _is_running_on_cpu():
      return fc._EmbeddingColumn._get_dense_tensor(
          self, inputs, weight_collections, trainable)

    # TPU mode
    # Get the embeddings from the LazyBuilder.
    tensor = inputs.get(self.get_feature_key_name())

    # Add to collection for _create_tpu_embedding_variables_and_ops
    _record_variable_scope_and_name(
        self.get_embedding_var_name(),
        'embedding_weights',
        bypass_scope_validation=self._bypass_scope_validation)

    return tensor

  def _get_sequence_dense_tensor(
      self, inputs, weight_collections=None, trainable=None):
    # [FUNCTION DOC]
    # Produces the sequence embedding result for this column. Inputs as for the dense variant.
    # Under TPU inference or on CPU it defers to the parent implementation (on the host via
    # outside compilation for inference). In TPU training mode it fetches the embedding tensor
    # and the sequence length tensor by feature key, removes the trailing singleton dimension
    # that the lazy builder added to the lengths, records the embedding variable name and scope
    # in a graph collection, and returns a pair of dense tensor and sequence length. Side
    # effect: mutates the graph collection; may raise a value error on a scope mismatch.
    if tpu.under_tpu_inference_context():
      def host_computation():
        # [FUNCTION DOC]
        # HELPER: host-side closure that calls the parent embedding column sequence logic with
        # the captured arguments and returns its result.
        return fc._EmbeddingColumn._get_sequence_dense_tensor(
            self, inputs, weight_collections, trainable)

      return tpu_replication.outside_compilation(host_computation)

    if _is_running_on_cpu():
      return fc._EmbeddingColumn._get_sequence_dense_tensor(
          self, inputs, weight_collections, trainable)

    tensor = inputs.get(self.get_feature_key_name())
    tensor_lengths = inputs.get(self.get_sequence_length_feature_key_name())

    # inputs is a _LazyBuilder and for rank 1 tensors, it calls expand_dims(-1).
    # We need to undo this to match the standard CPU sequence embedding.
    tensor_lengths = array_ops.squeeze(tensor_lengths, -1)

    # Add to collection for _create_tpu_embedding_variables_and_ops
    _record_variable_scope_and_name(
        self.get_embedding_var_name(),
        'embedding_weights',
        bypass_scope_validation=self._bypass_scope_validation)

    return fc._SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=tensor, sequence_length=tensor_lengths)


class _TPUSharedEmbeddingColumn(_TPUBaseEmbeddingColumn,
                                fc._SharedEmbeddingColumn):
  # [CLASS DOC]
  # HELPER: internal column type created by the shared embedding columns factory.
  # TPU variant of the core shared embedding column combining the TPU base mixin with the
  # parent shared column. Several instances share one embedding table identified by the shared
  # collection name. Provides accessors used when configuring TPU embedding tables and
  # overrides the dense and sequence tensor methods so that, in TPU mode, the lookup is done by
  # the TPU embedding engine and only the variable name and scope are recorded.
  """Core Shared Embedding Column."""

  def __new__(cls,
              categorical_column,
              dimension,
              combiner='mean',
              initializer=None,
              shared_embedding_collection_name=None,
              ckpt_to_load_from=None,
              tensor_name_in_ckpt=None,
              max_norm=None,
              trainable=True,
              max_sequence_length=0,
              learning_rate_fn=None,
              use_safe_embedding_lookup=True):
    # [FUNCTION DOC]
    # Allocation method. Forwards the embedding configuration (categorical column, dimension,
    # combiner, initializer, shared collection name, unsupported checkpoint and norm options,
    # trainable and safe lookup flags) to the parent shared embedding column allocator, using
    # this class as the type. The max sequence length and learning rate function are not
    # forwarded; they are handled by the initializer. Returns the new column instance.
    return fc._SharedEmbeddingColumn.__new__(
        cls,
        categorical_column,
        dimension,
        combiner=combiner,
        initializer=initializer,
        shared_embedding_collection_name=shared_embedding_collection_name,
        ckpt_to_load_from=ckpt_to_load_from,
        tensor_name_in_ckpt=tensor_name_in_ckpt,
        max_norm=max_norm,
        trainable=trainable,
        use_safe_embedding_lookup=use_safe_embedding_lookup)

  def __init__(self,
               categorical_column,
               dimension,
               combiner='mean',
               initializer=None,
               shared_embedding_collection_name=None,
               ckpt_to_load_from=None,
               tensor_name_in_ckpt=None,
               max_norm=None,
               trainable=True,
               max_sequence_length=0,
               learning_rate_fn=None,
               use_safe_embedding_lookup=True):
    # [FUNCTION DOC]
    # Initializer for the shared column. Takes the same arguments as the constructor-time
    # factory; only categorical_column, max_sequence_length and learning_rate_fn are used here
    # (passed to the base class initializer, which validates sequence length), and the key is
    # reset to null. The other arguments were consumed by the allocation method. Raises a value
    # error if the sequence length is inconsistent with the column kind.

    _TPUBaseEmbeddingColumn.__init__(
        self,
        categorical_column,
        max_sequence_length=max_sequence_length,
        learning_rate_fn=learning_rate_fn)
    self._key = None

  def get_combiner(self):
    # [FUNCTION DOC]
    # Returns the combiner name (how multiple ids in a row are reduced) stored on the column.
    return self.combiner

  def get_embedding_table_size(self):
    # [FUNCTION DOC]
    # Returns a pair of the number of buckets of the wrapped categorical column (vocabulary
    # size) and the embedding dimension.
    """Returns num_ids and width."""
    return (self.categorical_column._num_buckets, self.dimension)

  def get_feature_key_name(self):
    # [FUNCTION DOC]
    # Returns the name of the input feature that feeds this column: the inner categorical
    # column's name when the column is weighted, else the categorical column's own name.
    """get_feature_key_name."""
    if self.is_categorical_column_weighted():
      return self.categorical_column.categorical_column.name
    return self.categorical_column.name

  def get_weight_key_name(self):
    # [FUNCTION DOC]
    # Returns the feature key holding the per-id weights if the categorical column is weighted,
    # otherwise returns the null value.
    """get_weight_key_name."""
    if self.is_categorical_column_weighted():
      return self.categorical_column.weight_feature_key
    return None

  def get_embedding_var_name(self):
    # [FUNCTION DOC]
    # Returns the name of the embedding table variable. For shared columns this is the shared
    # collection name, so many columns map to one table.
    """get_embedding_var_name."""
    return self.shared_embedding_collection_name

  def get_initializer(self):
    # [FUNCTION DOC]
    # Returns the initializer function that was given to this shared column's constructor.
    return self.initializer

  def is_categorical_column_weighted(self):
    # [FUNCTION DOC]
    # Returns true if the wrapped categorical column is a weighted categorical column (old or
    # new style), otherwise false.
    """Check if the categorical column of the embedding column is weighted."""
    if isinstance(
        self.categorical_column,
        (
            fc._WeightedCategoricalColumn,  # pylint: disable=protected-access
            fc_lib.WeightedCategoricalColumn)):
      return True
    return False

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    # [FUNCTION DOC]
    # Produces the dense embedding tensor for this shared column. Inputs: inputs, the lazy
    # feature builder; weight_collections and trainable, passed to the parent on CPU paths.
    # Under TPU inference the parent computation is run on the host through outside
    # compilation; when running on CPU the parent implementation is used directly. In TPU
    # training mode it returns the already looked-up tensor fetched from inputs by feature key
    # and records the embedding variable name and scope (marked as shared) in a graph
    # collection for later table creation. Side effect: mutates that graph collection.
    if tpu.under_tpu_inference_context():
      def host_computation():
        # [FUNCTION DOC]
        # HELPER: host-side closure that calls the parent shared column dense tensor logic with
        # the captured inputs, weight collections and trainable flag; returns its tensor.
        return fc._SharedEmbeddingColumn._get_dense_tensor(
            self, inputs, weight_collections, trainable)

      return tpu_replication.outside_compilation(host_computation)

    if _is_running_on_cpu():
      return fc._SharedEmbeddingColumn._get_dense_tensor(
          self, inputs, weight_collections, trainable)

    # TPU mode
    # Get the embeddings from the LazyBuilder.
    tensor = inputs.get(self.get_feature_key_name())

    # Add to collection for _create_tpu_embedding_variables_and_ops
    _record_variable_scope_and_name(
        self.get_embedding_var_name(),
        'embedding_weights',
        is_shared_embedding=True)
    return tensor

  def _get_sequence_dense_tensor(
      self, inputs, weight_collections=None, trainable=None):
    # [FUNCTION DOC]
    # Produces the sequence embedding result for this shared column. Inputs as for the dense
    # variant. Under TPU inference or on CPU it defers to the parent implementation (on the
    # host via outside compilation for inference). In TPU training mode it fetches the
    # embedding tensor and the sequence length tensor by their feature keys and records the
    # shared embedding variable name and scope in a graph collection. Returns the parent's
    # result, or a pair of dense tensor and sequence length. Side effect: mutates the graph
    # collection.
    # KNOWN ISSUE: unlike the non-shared column, the TPU path does not squeeze the trailing
    # dimension of the length tensor, so lengths may keep a rank that differs from CPU output.
    if tpu.under_tpu_inference_context():
      def host_computation():
        # [FUNCTION DOC]
        # HELPER: host-side closure that calls the parent shared column sequence logic with the
        # captured arguments and returns its result.
        return fc._SharedEmbeddingColumn._get_sequence_dense_tensor(
            self, inputs, weight_collections, trainable)

      return tpu_replication.outside_compilation(host_computation)

    if _is_running_on_cpu():
      return fc._SharedEmbeddingColumn._get_sequence_dense_tensor(
          self, inputs, weight_collections, trainable)

    tensor = inputs.get(self.get_feature_key_name())
    tensor_lengths = inputs.get(self.get_sequence_length_feature_key_name())

    # Add to collection for _create_tpu_embedding_variables_and_ops
    _record_variable_scope_and_name(
        self.get_embedding_var_name(),
        'embedding_weights',
        is_shared_embedding=True)

    return fc._SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=tensor, sequence_length=tensor_lengths)


def _record_variable_scope_and_name(embedding_var_name,
                                    embedding_var_name_in_fc,
                                    is_shared_embedding=False,
                                    bypass_scope_validation=False):
  # [FUNCTION DOC]
  # HELPER: records, in a default-graph collection, which variable scope and column-internal
  # name each embedding variable uses, so TPU embedding tables can be created later.
  # Inputs: embedding_var_name, the table name; embedding_var_name_in_fc, the name used inside
  # the feature column; is_shared_embedding (default false) and bypass_scope_validation
  # (default false), either of which skips the scope-mismatch check.
  # Side effects: creates the collection entry (a mapping from name to scope name and inner
  # name) if missing, and adds a new entry if the name is unseen. Returns nothing.
  # Raises a value error if the name was seen before under a different variable scope (unless
  # skipped) or with a different inner name.
  """Add embedding variable name and scope to collection."""
  g = ops.get_default_graph()
  collection = g.get_collection_ref(_TPU_FC_TO_SCOPE)
  if not collection:
    collection.append({})

  var_def_dict = collection[0]

  captured_scope = variable_scope.get_variable_scope()
  captured_scope_name = captured_scope.name

  if embedding_var_name in var_def_dict:
    if (var_def_dict[embedding_var_name][0] != captured_scope_name and
        not is_shared_embedding and not bypass_scope_validation):
      raise ValueError(
          'For embedding var name {}, the variable scope name is different, '
          'got {}; expected {}'.format(embedding_var_name,
                                       captured_scope_name,
                                       var_def_dict[embedding_var_name][0]))
    if var_def_dict[embedding_var_name][1] != embedding_var_name_in_fc:
      raise ValueError(
          'For embedding var name {}, the embedding name is different, '
          'got {}; expected {}'.format(embedding_var_name,
                                       embedding_var_name_in_fc,
                                       var_def_dict[embedding_var_name][1]))
  else:
    var_def_dict[embedding_var_name] = (captured_scope_name,
                                        embedding_var_name_in_fc)


def _is_running_on_cpu():
  # [FUNCTION DOC]
  # HELPER: tells whether the current TPU context has no shards, meaning the model is built
  # for CPU execution. Takes no inputs; returns true when the number of shards is null,
  # otherwise false.
  """Returns True if the current context is CPU model."""
  return tpu_function.get_tpu_context().number_of_shards is None


def get_sequence_length_feature_key_name_from_feature_key_name(feature_name):
  # [FUNCTION DOC]
  # Input: feature_name, the key of a sequence feature (string). Returns the key of its
  # companion sequence length feature: the name with the sequence length suffix appended.
  # Pure; no side effects.
  """Gets the name of the sequence length feature from that of the base feature.

  Args:
    feature_name: The feature key of a sequence column.

  Returns:
    A string which is the feature key for the associated feature length column.
  """
  return feature_name + _SEQUENCE_FEATURE_LENGTH_POSTFIX
