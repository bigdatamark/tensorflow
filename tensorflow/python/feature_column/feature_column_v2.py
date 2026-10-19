# [MODULE DOC]
# Core definition of the version-2 FeatureColumn API used to describe how raw input features (dense,
# sparse, string, integer) are parsed from examples and transformed into tensors for models.
# Exposes the public factory functions (numeric_column, bucketized_column, embedding_column,
# shared_embedding_columns_v2, indicator_column, weighted_categorical_column, crossed_column and the
# categorical_column_with_* family, make_parse_example_spec_v2), the abstract base classes
# (DenseColumn, CategoricalColumn, SequenceDenseColumn), the concrete column classes (all immutable
# named tuples), the state management classes (StateManager and its implementation) and the
# FeatureTransformationCache that memoises transformed features while a model is built.
# Concrete columns also inherit from the legacy v1 column classes in feature_column.py so they can
# be used with both the old and new code paths; the underscore-prefixed methods are the deprecated
# v1 entry points, the non-prefixed ones are the v2 entry points.
# Import-time side effects: concrete column classes are registered with the feature-column
# serialization registry, and several symbols are exported to the public or internal TensorFlow API
# namespaces through the export decorators. No flags are defined and no global state is created
# besides the deprecation message constants.
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
"""This API defines FeatureColumn abstraction.

FeatureColumns provide a high level abstraction for ingesting and representing
features.

FeatureColumns can also be transformed into a generic input layer for
custom models using `input_layer`.

NOTE: Functions prefixed with "_" indicate experimental or private parts of
the API subject to change, and should not be relied upon!
"""

import abc
import collections
import math
import re

import numpy as np
import six

from tensorflow.python.data.experimental.ops import lookup_ops as data_lookup_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2_types as fc_types
from tensorflow.python.feature_column import serialization
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls

_FEATURE_COLUMN_DEPRECATION_DATE = None
_FEATURE_COLUMN_DEPRECATION = ('The old _FeatureColumn APIs are being '
                               'deprecated. Please use the new FeatureColumn '
                               'APIs instead.')
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


class StateManager(object):
  # [CLASS DOC]
  # Abstract interface through which feature columns create, store and look up variables and other
  # resources (lookup tables, trackables) without holding state themselves, since columns are meant
  # to be stateless configuration. Every method here is a stub that raises a not-implemented error;
  # concrete subclasses (for example the layer-backed implementation in this module) provide the
  # behavior. Holds no attributes of its own.
  """Manages the state associated with FeatureColumns.

  Some `FeatureColumn`s create variables or resources to assist their
  computation. The `StateManager` is responsible for creating and storing these
  objects since `FeatureColumn`s are supposed to be stateless configuration
  only.
  """

  def create_variable(self,
                      feature_column,
                      name,
                      shape,
                      dtype=None,
                      trainable=True,
                      use_resource=True,
                      initializer=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says dtype defaults to the object's dtype or float32, but this base
    # implementation ignores every argument and always raises.
    # Abstract: would create and return a new variable for the given feature column. Inputs: the
    # feature column the variable belongs to, a variable name, a shape, an optional data type, a
    # trainable flag, a flag choosing resource variables, and an optional initializer callable.
    # Discards all arguments and always raises a not-implemented error; subclasses must override.
    """Creates a new variable.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      name: variable name.
      shape: variable shape.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      trainable: Whether this variable is trainable or not.
      use_resource: If true, we use resource variables. Otherwise we use
        RefVariable.
      initializer: initializer instance (callable).

    Returns:
      The created variable.
    """
    del feature_column, name, shape, dtype, trainable, use_resource, initializer
    raise NotImplementedError('StateManager.create_variable')

  def add_variable(self, feature_column, var):
    # [FUNCTION DOC]
    # Abstract: would register an already existing variable (second argument) under the given feature
    # column (first argument). Discards both arguments and always raises a not-implemented error;
    # subclasses must override. Returns nothing.
    """Adds an existing variable to the state.

    Args:
      feature_column: A `FeatureColumn` object to associate this variable with.
      var: The variable.
    """
    del feature_column, var
    raise NotImplementedError('StateManager.add_variable')

  def get_variable(self, feature_column, name):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The raised error message names a method called get_var, which does not exist; the
    # actual method is get_variable.
    # Abstract: would return an existing variable identified by a feature column and a variable name.
    # Discards its arguments and always raises a not-implemented error; subclasses must override.
    """Returns an existing variable.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      name: variable name.
    """
    del feature_column, name
    raise NotImplementedError('StateManager.get_var')

  def add_resource(self, feature_column, name, resource):
    # [FUNCTION DOC]
    # Abstract: would store a resource (table, variable, trackable object, and so on) under a feature
    # column and a resource name and return the created resource. Discards all arguments and always
    # raises a not-implemented error; subclasses must override.
    """Creates a new resource.

    Resources can be things such as tables, variables, trackables, etc.

    Args:
      feature_column: A `FeatureColumn` object this resource corresponds to.
      name: Name of the resource.
      resource: The resource.

    Returns:
      The created resource.
    """
    del feature_column, name, resource
    raise NotImplementedError('StateManager.add_resource')

  def has_resource(self, feature_column, name):
    # [FUNCTION DOC]
    # Abstract: would report whether a resource with the given name already exists for the given
    # feature column. Discards its arguments and always raises a not-implemented error; subclasses
    # must override.
    """Returns true iff a resource with same name exists.

    Resources can be things such as tables, variables, trackables, etc.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      name: Name of the resource.
    """
    del feature_column, name
    raise NotImplementedError('StateManager.has_resource')

  def get_resource(self, feature_column, name):
    # [FUNCTION DOC]
    # Abstract: would return a previously stored resource identified by feature column and resource
    # name. Discards its arguments and always raises a not-implemented error; subclasses must override.
    """Returns an already created resource.

    Resources can be things such as tables, variables, trackables, etc.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      name: Name of the resource.
    """
    del feature_column, name
    raise NotImplementedError('StateManager.get_resource')


@tf_export('__internal__.feature_column.StateManager', v1=[])
class _StateManagerImpl(StateManager):
  # [CLASS DOC]
  # HELPER: Internal state manager used by the DenseFeatures and LinearModel style layers.
  # KNOWN ISSUE: The add_variable method of the base class is not overridden, so calling it on this
  # implementation still raises a not-implemented error.
  # Concrete StateManager that stores variables and resources on behalf of a Keras-style layer.
  # Holds the owning layer, a default trainable flag, a mapping from feature column to a mapping of
  # variable name to variable, and a mapping from feature column to a mapping of resource name to
  # resource. Variables are created through the layer's weight-creation method and tracked on the
  # layer; trackable resources are additionally published on the layer for checkpointing. Exported
  # as the internal feature-column state manager symbol.
  """Manages the state of DenseFeatures and LinearLayer.

  Some `FeatureColumn`s create variables or resources to assist their
  computation. The `StateManager` is responsible for creating and storing these
  objects since `FeatureColumn`s are supposed to be stateless configuration
  only.
  """

  def __init__(self, layer, trainable):
    # [FUNCTION DOC]
    # Creates the state manager. Inputs: the layer this manager is attached to (may be nothing) and a
    # default trainable flag that is combined (logical and) with the per-variable flag. Side effects:
    # when a layer is given and has no resources mapping yet, attaches an empty tracked mapping named
    # for resources to the layer; initialises the two per-column dictionaries (variables, resources)
    # as auto-creating mappings. Returns nothing.
    """Creates an _StateManagerImpl object.

    Args:
      layer: The input layer this state manager is associated with.
      trainable: Whether by default, variables created are trainable or not.
    """
    self._trainable = trainable
    self._layer = layer
    if self._layer is not None and not hasattr(self._layer, '_resources'):
      self._layer._resources = data_structures.Mapping()  # pylint: disable=protected-access
    self._cols_to_vars_map = collections.defaultdict(lambda: {})
    self._cols_to_resources_map = collections.defaultdict(lambda: {})

  def create_variable(self,
                      feature_column,
                      name,
                      shape,
                      dtype=None,
                      trainable=True,
                      use_resource=True,
                      initializer=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring claims dtype defaults to the object's dtype or float32, but the value
    # is passed through unchanged (possibly nothing) to the layer, which decides the default.
    # Creates a new weight on the owning layer for the given feature column and records it. Inputs:
    # feature column, variable name, shape, optional dtype, a trainable flag (effective trainability
    # is this flag and the manager's flag), use_resource flag and optional initializer. Raises a
    # value error if a variable of that name already exists for that column. Side effects: calls the
    # layer's add_weight with the global variable-scope getter inside a scope that disables the
    # layer's automatic dependency tracking, then explicitly tracks the variable (each partition of a
    # partitioned variable, or the single variable if it is trackable) on the layer under a name built
    # from the column name and variable name, and stores it in the variables mapping. Returns the new
    # variable (possibly partitioned).
    """Creates a new variable.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      name: variable name.
      shape: variable shape.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      trainable: Whether this variable is trainable or not.
      use_resource: If true, we use resource variables. Otherwise we use
        RefVariable.
      initializer: initializer instance (callable).

    Returns:
      The created variable.
    """
    if name in self._cols_to_vars_map[feature_column]:
      raise ValueError('Variable already exists.')

    # We explicitly track these variables since `name` is not guaranteed to be
    # unique and disable manual tracking that the add_weight call does.
    with trackable.no_manual_dependency_tracking_scope(self._layer):
      var = self._layer.add_weight(
          name=name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          trainable=self._trainable and trainable,
          use_resource=use_resource,
          # TODO(rohanj): Get rid of this hack once we have a mechanism for
          # specifying a default partitioner for an entire layer. In that case,
          # the default getter for Layers should work.
          getter=variable_scope.get_variable)
    if isinstance(var, variables.PartitionedVariable):
      for v in var:
        part_name = name + '/' + str(v._get_save_slice_info().var_offset[0])  # pylint: disable=protected-access
        self._layer._track_trackable(v, feature_column.name + '/' + part_name)  # pylint: disable=protected-access
    else:
      if isinstance(var, trackable.Trackable):
        self._layer._track_trackable(var, feature_column.name + '/' + name)  # pylint: disable=protected-access

    self._cols_to_vars_map[feature_column][name] = var
    return var

  def get_variable(self, feature_column, name):
    # [FUNCTION DOC]
    # Returns the variable previously created for the given feature column and variable name. Raises a
    # value error saying the variable does not exist if there is none. Read-only lookup, but note the
    # per-column mapping is auto-created on access for an unknown column.
    """Returns an existing variable.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      name: variable name.
    """
    if name in self._cols_to_vars_map[feature_column]:
      return self._cols_to_vars_map[feature_column][name]
    raise ValueError('Variable does not exist.')

  def add_resource(self, feature_column, resource_name, resource):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says it returns the created resource, but the method returns nothing.
    # Stores a resource (table, variable, trackable) under the given feature column and resource name.
    # If the manager has a layer and the resource is trackable, also publishes it in the layer's
    # resource mapping keyed first by column name and then by resource name (creating the inner tracked
    # mapping when missing, and not overwriting an entry that already exists there) so it is saved with
    # checkpoints.
    """Creates a new resource.

    Resources can be things such as tables, variables, trackables, etc.

    Args:
      feature_column: A `FeatureColumn` object this resource corresponds to.
      resource_name: Name of the resource.
      resource: The resource.

    Returns:
      The created resource.
    """
    self._cols_to_resources_map[feature_column][resource_name] = resource
    # pylint: disable=protected-access
    if self._layer is not None and isinstance(resource, trackable.Trackable):
      # Add trackable resources to the layer for serialization.
      if feature_column.name not in self._layer._resources:
        self._layer._resources[feature_column.name] = data_structures.Mapping()
      if resource_name not in self._layer._resources[feature_column.name]:
        self._layer._resources[feature_column.name][resource_name] = resource
    # pylint: enable=protected-access

  def has_resource(self, feature_column, resource_name):
    # [FUNCTION DOC]
    # Returns true when a resource with the given name has been stored for the given feature column,
    # false otherwise. Has the side effect of creating an empty per-column mapping for an unseen column.
    """Returns true iff a resource with same name exists.

    Resources can be things such as tables, variables, trackables, etc.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      resource_name: Name of the resource.
    """
    return resource_name in self._cols_to_resources_map[feature_column]

  def get_resource(self, feature_column, resource_name):
    # [FUNCTION DOC]
    # Returns the resource stored for the given feature column and resource name. Raises a value
    # error saying the resource does not exist when the column is unknown or the name is missing.
    """Returns an already created resource.

    Resources can be things such as tables, variables, trackables, etc.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      resource_name: Name of the resource.
    """
    if (feature_column not in self._cols_to_resources_map or
        resource_name not in self._cols_to_resources_map[feature_column]):
      raise ValueError('Resource does not exist.')
    return self._cols_to_resources_map[feature_column][resource_name]


def _transform_features_v2(features, feature_columns, state_manager):
  # [FUNCTION DOC]
  # HELPER: Internal utility that evaluates a set of feature columns against raw features; used by
  # higher-level layers and tests rather than called directly by users.
  # Inputs: a mapping from feature keys to tensors or sparse tensors, an iterable of feature columns,
  # and a state manager holding column state. Normalises the columns (validates types, rejects
  # duplicates and empty input, sorts by name), creates one transformation cache for the features,
  # and for each column evaluates it under a name scope derived from a sanitised column name.
  # Returns a dictionary mapping each feature column to its transformed tensor or sparse tensor.
  # Raises a value error for invalid column lists. Side effects: graph name scopes, and any
  # resources the columns create through the state manager.
  """Returns transformed features based on features columns passed in.

  Please note that most probably you would not need to use this function. Please
  check `input_layer` and `linear_model` to see whether they will
  satisfy your use case or not.

  Example:

  ```python
  # Define features and transformations
  crosses_a_x_b = crossed_column(
      columns=["sparse_feature_a", "sparse_feature_b"], hash_bucket_size=10000)
  price_buckets = bucketized_column(
      source_column=numeric_column("price"), boundaries=[...])

  columns = [crosses_a_x_b, price_buckets]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  transformed = transform_features(features=features, feature_columns=columns)

  assertCountEqual(columns, transformed.keys())
  ```

  Args:
    features: A mapping from key to tensors. `FeatureColumn`s look up via these
      keys. For example `numeric_column('price')` will look at 'price' key in
      this dict. Values can be a `SparseTensor` or a `Tensor` depends on
      corresponding `FeatureColumn`.
    feature_columns: An iterable containing all the `FeatureColumn`s.
    state_manager: A StateManager object that holds the FeatureColumn state.

  Returns:
    A `dict` mapping `FeatureColumn` to `Tensor` and `SparseTensor` values.
  """
  feature_columns = _normalize_feature_columns(feature_columns)
  outputs = {}
  with ops.name_scope(
      None, default_name='transform_features', values=features.values()):
    transformation_cache = FeatureTransformationCache(features)
    for column in feature_columns:
      with ops.name_scope(
          None,
          default_name=_sanitize_column_name_for_variable_scope(column.name)):
        outputs[column] = transformation_cache.get(column, state_manager)
  return outputs


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export(
    'feature_column.make_parse_example_spec',
    v1=[])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def make_parse_example_spec_v2(feature_columns):
  # [FUNCTION DOC]
  # Public API (exported as feature_column.make_parse_example_spec, deprecated at runtime).
  # Builds the parsing specification dictionary for example parsing from an iterable of feature
  # columns. Each column contributes its parse-spec mapping (feature key to fixed-length or
  # variable-length feature description); the results are merged. Raises a value error when an
  # item is not a FeatureColumn instance, or when two columns demand different specs for the same
  # key. Returns the merged mapping; no other side effects.
  """Creates parsing spec dictionary from input feature_columns.

  The returned dictionary can be used as arg 'features' in
  `tf.io.parse_example`.

  Typical usage example:

  ```python
  # Define features and transformations
  feature_a = tf.feature_column.categorical_column_with_vocabulary_file(...)
  feature_b = tf.feature_column.numeric_column(...)
  feature_c_bucketized = tf.feature_column.bucketized_column(
      tf.feature_column.numeric_column("feature_c"), ...)
  feature_a_x_feature_c = tf.feature_column.crossed_column(
      columns=["feature_a", feature_c_bucketized], ...)

  feature_columns = set(
      [feature_b, feature_c_bucketized, feature_a_x_feature_c])
  features = tf.io.parse_example(
      serialized=serialized_examples,
      features=tf.feature_column.make_parse_example_spec(feature_columns))
  ```

  For the above example, make_parse_example_spec would return the dict:

  ```python
  {
      "feature_a": parsing_ops.VarLenFeature(tf.string),
      "feature_b": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
      "feature_c": parsing_ops.FixedLenFeature([1], dtype=tf.float32)
  }
  ```

  Args:
    feature_columns: An iterable containing all feature columns. All items
      should be instances of classes derived from `FeatureColumn`.

  Returns:
    A dict mapping each feature key to a `FixedLenFeature` or `VarLenFeature`
    value.

  Raises:
    ValueError: If any of the given `feature_columns` is not a `FeatureColumn`
      instance.
  """
  result = {}
  for column in feature_columns:
    if not isinstance(column, fc_types.FeatureColumn):
      raise ValueError('All feature_columns must be FeatureColumn instances. '
                       'Given: {}'.format(column))
    config = column.parse_example_spec
    for key, value in six.iteritems(config):
      if key in result and value != result[key]:
        raise ValueError('feature_columns contain different parse_spec for key '
                         '{}. Given {} and {}'.format(key, value, result[key]))
    result.update(config)
  return result


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.embedding_column')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def embedding_column(categorical_column,
                     dimension,
                     combiner='mean',
                     initializer=None,
                     ckpt_to_load_from=None,
                     tensor_name_in_ckpt=None,
                     max_norm=None,
                     trainable=True,
                     use_safe_embedding_lookup=True):
  # [FUNCTION DOC]
  # KNOWN ISSUE: The docstring lists a runtime error when eager execution is enabled, but this
  # function never checks for eager execution and never raises it.
  # Public API (deprecated at runtime). Factory that wraps a categorical column into a dense
  # embedding column. Inputs: the categorical column; embedding dimension (positive integer); combiner
  # name for multi-valued rows (default mean; sqrtn and sum also supported); optional initializer
  # callable (default truncated normal with mean zero and standard deviation one over the square
  # root of the dimension); optional checkpoint path and tensor name to restore weights from (must be
  # given together or both omitted); optional max norm for clipping; trainable flag (default true);
  # and a flag choosing the safe sparse embedding lookup (default true). Raises a value error for a
  # missing or non-positive dimension, for exactly one of the two checkpoint arguments being given,
  # or for a non-callable initializer. Returns a new EmbeddingColumn; creates no variables itself.
  """`DenseColumn` that converts from sparse, categorical input.

  Use this when your inputs are sparse, but you want to convert them to a dense
  representation (e.g., to feed to a DNN).

  Args:
    categorical_column: A `CategoricalColumn` created by a
      `categorical_column_with_*` function. This column produces the sparse IDs
      that are inputs to the embedding lookup.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from which
      to restore the column weights. Required if `ckpt_to_load_from` is not
      `None`.
    max_norm: If not `None`, embedding values are l2-normalized to this value.
    trainable: Whether or not the embedding is trainable. Default is True.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies to rank 2 (NxM) shaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.

  Returns:
    `DenseColumn` that converts from sparse input.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: If eager execution is enabled.
  """
  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified. '
                     'Embedding of column_name: {}'.format(
                         categorical_column.name))
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1 / math.sqrt(dimension))

  return EmbeddingColumn(
      categorical_column=categorical_column,
      dimension=dimension,
      combiner=combiner,
      initializer=initializer,
      ckpt_to_load_from=ckpt_to_load_from,
      tensor_name_in_ckpt=tensor_name_in_ckpt,
      max_norm=max_norm,
      trainable=trainable,
      use_safe_embedding_lookup=use_safe_embedding_lookup)


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export(v1=['feature_column.shared_embedding_columns'])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def shared_embedding_columns(categorical_columns,
                             dimension,
                             combiner='mean',
                             initializer=None,
                             shared_embedding_collection_name=None,
                             ckpt_to_load_from=None,
                             tensor_name_in_ckpt=None,
                             max_norm=None,
                             trainable=True,
                             use_safe_embedding_lookup=True):
  # [FUNCTION DOC]
  # KNOWN ISSUE: The mismatching-bucket error message contains the typo "ven column" instead of
  # "Given column".
  # Public v1-only API (deprecated at runtime) that returns embedding columns sharing one weight
  # matrix, implemented with the legacy shared embedding column class. Inputs: a list of categorical
  # columns (same type, same bucket count, weighted/sequence wrappers allowed), dimension, combiner,
  # initializer, optional shared collection name (derived from the sorted column names when absent),
  # optional checkpoint path and tensor name (both or neither), max norm, trainable flag and
  # safe-lookup flag. Raises a runtime error when eager execution is enabled; value errors for a bad
  # dimension, a lone checkpoint argument, a non-callable initializer, columns that are not legacy
  # categorical columns, differing column types, or differing bucket counts. Returns a list of shared
  # embedding columns in the order of the input columns.
  """List of dense columns that convert from sparse, categorical input.

  This is similar to `embedding_column`, except that it produces a list of
  embedding columns that share the same embedding weights.

  Use this when your inputs are sparse and of the same type (e.g. watched and
  impression video IDs that share the same vocabulary), and you want to convert
  them to a dense representation (e.g., to feed to a DNN).

  Inputs must be a list of categorical columns created by any of the
  `categorical_column_*` function. They must all be of the same type and have
  the same arguments except `key`. E.g. they can be
  categorical_column_with_vocabulary_file with the same vocabulary_file. Some or
  all columns could also be weighted_categorical_column.

  Args:
    categorical_columns: List of categorical columns created by a
      `categorical_column_with_*` function. These columns produce the sparse IDs
      that are inputs to the embedding lookup. All columns must be of the same
      type and have the same arguments except `key`. E.g. they can be
      categorical_column_with_vocabulary_file with the same vocabulary_file.
      Some or all columns could also be weighted_categorical_column.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional name of the collection where
      shared embedding weights are added. If not given, a reasonable name will
      be chosen based on the names of `categorical_columns`. This is also used
      in `variable_scope` when creating shared embedding weights.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from which
      to restore the column weights. Required if `ckpt_to_load_from` is not
      `None`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    trainable: Whether or not the embedding is trainable. Default is True.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies to rank 2 (NxM) shaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.

  Returns:
    A list of dense columns that converts from sparse input. The order of
    results follows the ordering of `categorical_columns`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if any of the given `categorical_columns` is of different type
      or has different arguments than the others.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: if eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('shared_embedding_columns are not supported when eager '
                       'execution is enabled.')

  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified.')
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1. / math.sqrt(dimension))

  # Sort the columns so the default collection name is deterministic even if the
  # user passes columns from an unsorted collection, such as dict.values().
  sorted_columns = sorted(categorical_columns, key=lambda x: x.name)

  c0 = sorted_columns[0]
  num_buckets = c0._num_buckets  # pylint: disable=protected-access
  if not isinstance(c0, fc_old._CategoricalColumn):  # pylint: disable=protected-access
    raise ValueError(
        'All categorical_columns must be subclasses of _CategoricalColumn. '
        'Given: {}, of type: {}'.format(c0, type(c0)))
  while isinstance(
      c0,
      (
          fc_old._WeightedCategoricalColumn,  # pylint: disable=protected-access
          WeightedCategoricalColumn,
          fc_old._SequenceCategoricalColumn,  # pylint: disable=protected-access
          SequenceCategoricalColumn)):
    c0 = c0.categorical_column
  for c in sorted_columns[1:]:
    while isinstance(
        c,
        (
            fc_old._WeightedCategoricalColumn,  # pylint: disable=protected-access
            WeightedCategoricalColumn,
            fc_old._SequenceCategoricalColumn,  # pylint: disable=protected-access
            SequenceCategoricalColumn)):
      c = c.categorical_column
    if not isinstance(c, type(c0)):
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same type, or be weighted_categorical_column or sequence column '
          'of the same type. Given column: {} of type: {} does not match given '
          'column: {} of type: {}'.format(c0, type(c0), c, type(c)))
    if num_buckets != c._num_buckets:  # pylint: disable=protected-access
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same number of buckets. ven column: {} with buckets: {} does  '
          'not match column: {} with buckets: {}'.format(
              c0, num_buckets, c, c._num_buckets))  # pylint: disable=protected-access

  if not shared_embedding_collection_name:
    shared_embedding_collection_name = '_'.join(c.name for c in sorted_columns)
    shared_embedding_collection_name += '_shared_embedding'

  result = []
  for column in categorical_columns:
    result.append(
        fc_old._SharedEmbeddingColumn(  # pylint: disable=protected-access
            categorical_column=column,
            initializer=initializer,
            dimension=dimension,
            combiner=combiner,
            shared_embedding_collection_name=shared_embedding_collection_name,
            ckpt_to_load_from=ckpt_to_load_from,
            tensor_name_in_ckpt=tensor_name_in_ckpt,
            max_norm=max_norm,
            trainable=trainable,
            use_safe_embedding_lookup=use_safe_embedding_lookup))

  return result


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export(
    'feature_column.shared_embeddings',
    v1=[])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def shared_embedding_columns_v2(categorical_columns,
                                dimension,
                                combiner='mean',
                                initializer=None,
                                shared_embedding_collection_name=None,
                                ckpt_to_load_from=None,
                                tensor_name_in_ckpt=None,
                                max_norm=None,
                                trainable=True,
                                use_safe_embedding_lookup=True):
  # [FUNCTION DOC]
  # Public API (exported as feature_column.shared_embeddings, deprecated at runtime). Same argument
  # validation as the v1 version but builds a SharedEmbeddingColumnCreator holding the dimension,
  # initializer, checkpoint information, bucket count, trainable flag, collection name and safe-lookup
  # flag, and calls it once per input column. Raises a runtime error when eager execution is enabled;
  # value errors for a bad dimension, a lone checkpoint argument, a non-callable initializer, columns
  # not derived from the CategoricalColumn base class, differing column types (after unwrapping
  # weighted and sequence wrappers) or differing bucket counts. Returns a list of shared embedding
  # columns ordered like the input. The shared weight variable is created lazily later.
  """List of dense columns that convert from sparse, categorical input.

  This is similar to `embedding_column`, except that it produces a list of
  embedding columns that share the same embedding weights.

  Use this when your inputs are sparse and of the same type (e.g. watched and
  impression video IDs that share the same vocabulary), and you want to convert
  them to a dense representation (e.g., to feed to a DNN).

  Inputs must be a list of categorical columns created by any of the
  `categorical_column_*` function. They must all be of the same type and have
  the same arguments except `key`. E.g. they can be
  categorical_column_with_vocabulary_file with the same vocabulary_file. Some or
  all columns could also be weighted_categorical_column.

  Args:
    categorical_columns: List of categorical columns created by a
      `categorical_column_with_*` function. These columns produce the sparse IDs
      that are inputs to the embedding lookup. All columns must be of the same
      type and have the same arguments except `key`. E.g. they can be
      categorical_column_with_vocabulary_file with the same vocabulary_file.
      Some or all columns could also be weighted_categorical_column.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional collective name of these columns.
      If not given, a reasonable name will be chosen based on the names of
      `categorical_columns`.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from which
      to restore the column weights. Required if `ckpt_to_load_from` is not
      `None`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    trainable: Whether or not the embedding is trainable. Default is True.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies to rank 2 (NxM) shaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.

  Returns:
    A list of dense columns that converts from sparse input. The order of
    results follows the ordering of `categorical_columns`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if any of the given `categorical_columns` is of different type
      or has different arguments than the others.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: if eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('shared_embedding_columns are not supported when eager '
                       'execution is enabled.')

  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified.')
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1. / math.sqrt(dimension))

  # Sort the columns so the default collection name is deterministic even if the
  # user passes columns from an unsorted collection, such as dict.values().
  sorted_columns = sorted(categorical_columns, key=lambda x: x.name)

  c0 = sorted_columns[0]
  num_buckets = c0.num_buckets
  if not isinstance(c0, CategoricalColumn):
    raise ValueError(
        'All categorical_columns must be subclasses of CategoricalColumn. '
        'Given: {}, of type: {}'.format(c0, type(c0)))
  while isinstance(c0, (WeightedCategoricalColumn, SequenceCategoricalColumn)):
    c0 = c0.categorical_column
  for c in sorted_columns[1:]:
    while isinstance(c, (WeightedCategoricalColumn, SequenceCategoricalColumn)):
      c = c.categorical_column
    if not isinstance(c, type(c0)):
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same type, or be weighted_categorical_column or sequence column '
          'of the same type. Given column: {} of type: {} does not match given '
          'column: {} of type: {}'.format(c0, type(c0), c, type(c)))
    if num_buckets != c.num_buckets:
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same number of buckets. Given column: {} with buckets: {} does  '
          'not match column: {} with buckets: {}'.format(
              c0, num_buckets, c, c.num_buckets))

  if not shared_embedding_collection_name:
    shared_embedding_collection_name = '_'.join(c.name for c in sorted_columns)
    shared_embedding_collection_name += '_shared_embedding'

  column_creator = SharedEmbeddingColumnCreator(
      dimension, initializer, ckpt_to_load_from, tensor_name_in_ckpt,
      num_buckets, trainable, shared_embedding_collection_name,
      use_safe_embedding_lookup)

  result = []
  for column in categorical_columns:
    result.append(
        column_creator(
            categorical_column=column, combiner=combiner, max_norm=max_norm))

  return result


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.numeric_column')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def numeric_column(key,
                   shape=(1,),
                   default_value=None,
                   dtype=dtypes.float32,
                   normalizer_fn=None):
  # [FUNCTION DOC]
  # Public API (deprecated at runtime) factory for real-valued features. Inputs: a string key (also
  # the column name); a shape (integer or iterable of positive integers, default one element); a
  # default value (single value or one per element, used when parsing finds data missing; none means
  # parsing fails on missing data); a dtype (integer or floating, default float32); and an optional
  # normalizer callable applied to the input tensor. Validates the shape, the dtype, the default
  # value against shape and dtype, the callability of the normalizer and that the key is a string.
  # Raises type errors for non-integer shape dimensions or incompatible defaults or a non-callable
  # normalizer, and value errors for non-positive dimensions or a non-numeric dtype. Returns a
  # NumericColumn.
  """Represents real valued or numerical features.

  Example:

  Assume we have data with two features `a` and `b`.

  >>> data = {'a': [15, 9, 17, 19, 21, 18, 25, 30],
  ...    'b': [5.0, 6.4, 10.5, 13.6, 15.7, 19.9, 20.3 , 0.0]}

  Let us represent the features `a` and `b` as numerical features.

  >>> a = tf.feature_column.numeric_column('a')
  >>> b = tf.feature_column.numeric_column('b')

  Feature column describe a set of transformations to the inputs.

  For example, to "bucketize" feature `a`, wrap the `a` column in a
  `feature_column.bucketized_column`.
  Providing `5` bucket boundaries, the bucketized_column api
  will bucket this feature in total of `6` buckets.

  >>> a_buckets = tf.feature_column.bucketized_column(a,
  ...    boundaries=[10, 15, 20, 25, 30])

  Create a `DenseFeatures` layer which will apply the transformations
  described by the set of `tf.feature_column` objects:

  >>> feature_layer = tf.keras.layers.DenseFeatures([a_buckets, b])
  >>> print(feature_layer(data))
  tf.Tensor(
  [[ 0.   0.   1.   0.   0.   0.   5. ]
   [ 1.   0.   0.   0.   0.   0.   6.4]
   [ 0.   0.   1.   0.   0.   0.  10.5]
   [ 0.   0.   1.   0.   0.   0.  13.6]
   [ 0.   0.   0.   1.   0.   0.  15.7]
   [ 0.   0.   1.   0.   0.   0.  19.9]
   [ 0.   0.   0.   0.   1.   0.  20.3]
   [ 0.   0.   0.   0.   0.   1.   0. ]], shape=(8, 7), dtype=float32)

  Args:
    key: A unique string identifying the input feature. It is used as the column
      name and the dictionary key for feature parsing configs, feature `Tensor`
      objects, and feature columns.
    shape: An iterable of integers specifies the shape of the `Tensor`. An
      integer can be given which means a single dimension `Tensor` with given
      width. The `Tensor` representing the column will have the shape of
      [batch_size] + `shape`.
    default_value: A single value compatible with `dtype` or an iterable of
      values compatible with `dtype` which the column takes on during
      `tf.Example` parsing if data is missing. A default value of `None` will
      cause `tf.io.parse_example` to fail if an example does not contain this
      column. If a single value is provided, the same value will be applied as
      the default value for every item. If an iterable of values is provided,
      the shape of the `default_value` should be equal to the given `shape`.
    dtype: defines the type of values. Default value is `tf.float32`. Must be a
      non-quantized, real integer or floating point type.
    normalizer_fn: If not `None`, a function that can be used to normalize the
      value of the tensor after `default_value` is applied for parsing.
      Normalizer function takes the input `Tensor` as its argument, and returns
      the output `Tensor`. (e.g. lambda x: (x - 3.0) / 4.2). Please note that
      even though the most common use case of this function is normalization, it
      can be used for any kind of Tensorflow transformations.

  Returns:
    A `NumericColumn`.

  Raises:
    TypeError: if any dimension in shape is not an int
    ValueError: if any dimension in shape is not a positive integer
    TypeError: if `default_value` is an iterable but not compatible with `shape`
    TypeError: if `default_value` is not compatible with `dtype`.
    ValueError: if `dtype` is not convertible to `tf.float32`.
  """
  shape = _check_shape(shape, key)
  if not (dtype.is_integer or dtype.is_floating):
    raise ValueError('dtype must be convertible to float. '
                     'dtype: {}, key: {}'.format(dtype, key))
  default_value = fc_utils.check_default_value(shape, default_value, dtype, key)

  if normalizer_fn is not None and not callable(normalizer_fn):
    raise TypeError(
        'normalizer_fn must be a callable. Given: {}'.format(normalizer_fn))

  fc_utils.assert_key_is_string(key)
  return NumericColumn(
      key,
      shape=shape,
      default_value=default_value,
      dtype=dtype,
      normalizer_fn=normalizer_fn)


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.bucketized_column')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def bucketized_column(source_column, boundaries):
  # [FUNCTION DOC]
  # Public API (deprecated at runtime) that discretises a one-dimensional numeric column into buckets
  # defined by sorted boundaries (left boundary inclusive, right exclusive, so n boundaries give n+1
  # buckets). Inputs: the source numeric column (v1 or v2) and a non-empty strictly increasing list
  # or tuple of boundaries. Raises a value error when the source is not a numeric column, has more than
  # one dimension, or the boundaries are empty, not a list or tuple, or not strictly increasing.
  # Returns a BucketizedColumn with the boundaries stored as a tuple.
  """Represents discretized dense input bucketed by `boundaries`.

  Buckets include the left boundary, and exclude the right boundary. Namely,
  `boundaries=[0., 1., 2.]` generates buckets `(-inf, 0.)`, `[0., 1.)`,
  `[1., 2.)`, and `[2., +inf)`.

  For example, if the inputs are

  ```python
  boundaries = [0, 10, 100]
  input tensor = [[-5, 10000]
                  [150,   10]
                  [5,    100]]
  ```

  then the output will be

  ```python
  output = [[0, 3]
            [3, 2]
            [1, 3]]
  ```

  Example:

  ```python
  price = tf.feature_column.numeric_column('price')
  bucketized_price = tf.feature_column.bucketized_column(
      price, boundaries=[...])
  columns = [bucketized_price, ...]
  features = tf.io.parse_example(
      ..., features=tf.feature_column.make_parse_example_spec(columns))
  dense_tensor = tf.keras.layers.DenseFeatures(columns)(features)
  ```

  A `bucketized_column` can also be crossed with another categorical column
  using `crossed_column`:

  ```python
  price = tf.feature_column.numeric_column('price')
  # bucketized_column converts numerical feature to a categorical one.
  bucketized_price = tf.feature_column.bucketized_column(
      price, boundaries=[...])
  # 'keywords' is a string feature.
  price_x_keywords = tf.feature_column.crossed_column(
      [bucketized_price, 'keywords'], 50K)
  columns = [price_x_keywords, ...]
  features = tf.io.parse_example(
      ..., features=tf.feature_column.make_parse_example_spec(columns))
  dense_tensor = tf.keras.layers.DenseFeatures(columns)(features)
  linear_model = tf.keras.experimental.LinearModel(units=...)(dense_tensor)
  ```

  Args:
    source_column: A one-dimensional dense column which is generated with
      `numeric_column`.
    boundaries: A sorted list or tuple of floats specifying the boundaries.

  Returns:
    A `BucketizedColumn`.

  Raises:
    ValueError: If `source_column` is not a numeric column, or if it is not
      one-dimensional.
    ValueError: If `boundaries` is not a sorted list or tuple.
  """
  if not isinstance(source_column, (NumericColumn, fc_old._NumericColumn)):  # pylint: disable=protected-access
    raise ValueError(
        'source_column must be a column generated with numeric_column(). '
        'Given: {}'.format(source_column))
  if len(source_column.shape) > 1:
    raise ValueError('source_column must be one-dimensional column. '
                     'Given: {}'.format(source_column))
  if not boundaries:
    raise ValueError('boundaries must not be empty.')
  if not (isinstance(boundaries, list) or isinstance(boundaries, tuple)):
    raise ValueError('boundaries must be a sorted list.')
  for i in range(len(boundaries) - 1):
    if boundaries[i] >= boundaries[i + 1]:
      raise ValueError('boundaries must be a sorted list.')
  return BucketizedColumn(source_column, tuple(boundaries))


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.categorical_column_with_hash_bucket')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def categorical_column_with_hash_bucket(key,
                                        hash_bucket_size,
                                        dtype=dtypes.string):
  # [FUNCTION DOC]
  # KNOWN ISSUE: Docs say the bucket size must be an integer greater than 1, but the code accepts 1
  # (it only rejects values below 1, with a message saying at least 1).
  # Public API (deprecated at runtime) factory for a categorical column whose ids come from hashing
  # string values (integers are converted to strings first). Inputs: string key, number of hash
  # buckets, and dtype (string by default; only string and integer types allowed). Raises a value
  # error when the bucket size is missing or below 1 or the dtype is neither string nor integer, and
  # an error when the key is not a string. Returns a HashedCategoricalColumn.
  """Represents sparse feature where ids are set by hashing.

  Use this when your sparse features are in string or integer format, and you
  want to distribute your inputs into a finite number of buckets by hashing.
  output_id = Hash(input_feature_string) % bucket_size for string type input.
  For int type input, the value is converted to its string representation first
  and then hashed by the same formula.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example:

  ```python
  import tensorflow as tf
  keywords = tf.feature_column.categorical_column_with_hash_bucket("keywords",
  10000)
  columns = [keywords]
  features = {'keywords': tf.constant([['Tensorflow', 'Keras', 'RNN', 'LSTM',
  'CNN'], ['LSTM', 'CNN', 'Tensorflow', 'Keras', 'RNN'], ['CNN', 'Tensorflow',
  'LSTM', 'Keras', 'RNN']])}
  linear_prediction, _, _ = tf.compat.v1.feature_column.linear_model(features,
  columns)

  # or
  import tensorflow as tf
  keywords = tf.feature_column.categorical_column_with_hash_bucket("keywords",
  10000)
  keywords_embedded = tf.feature_column.embedding_column(keywords, 16)
  columns = [keywords_embedded]
  features = {'keywords': tf.constant([['Tensorflow', 'Keras', 'RNN', 'LSTM',
  'CNN'], ['LSTM', 'CNN', 'Tensorflow', 'Keras', 'RNN'], ['CNN', 'Tensorflow',
  'LSTM', 'Keras', 'RNN']])}
  input_layer = tf.keras.layers.DenseFeatures(columns)
  dense_tensor = input_layer(features)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the column
      name and the dictionary key for feature parsing configs, feature `Tensor`
      objects, and feature columns.
    hash_bucket_size: An int > 1. The number of buckets.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `HashedCategoricalColumn`.

  Raises:
    ValueError: `hash_bucket_size` is not greater than 1.
    ValueError: `dtype` is neither string nor integer.
  """
  if hash_bucket_size is None:
    raise ValueError('hash_bucket_size must be set. ' 'key: {}'.format(key))

  if hash_bucket_size < 1:
    raise ValueError('hash_bucket_size must be at least 1. '
                     'hash_bucket_size: {}, key: {}'.format(
                         hash_bucket_size, key))

  fc_utils.assert_key_is_string(key)
  fc_utils.assert_string_or_int(dtype, prefix='column_name: {}'.format(key))

  return HashedCategoricalColumn(key, hash_bucket_size, dtype)


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export(v1=['feature_column.categorical_column_with_vocabulary_file'])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def categorical_column_with_vocabulary_file(key,
                                            vocabulary_file,
                                            vocabulary_size=None,
                                            num_oov_buckets=0,
                                            default_value=None,
                                            dtype=dtypes.string):
  # [FUNCTION DOC]
  # Public v1-only API (deprecated at runtime) for a categorical column backed by a vocabulary file.
  # Thin wrapper that forwards to the v2 function, reordering the arguments (the v2 order puts dtype
  # and default value before the out-of-vocabulary bucket count). Inputs: key, vocabulary file path,
  # optional vocabulary size (inferred from the file when absent), number of out-of-vocabulary
  # buckets (default zero), optional default id for unknown values, and dtype (default string).
  # Returns the column produced by the v2 function and raises the same value errors.
  """A `CategoricalColumn` with a vocabulary file.

  Use this when your inputs are in string or integer format, and you have a
  vocabulary file that maps each value to an integer ID. By default,
  out-of-vocabulary values are ignored. Use either (but not both) of
  `num_oov_buckets` and `default_value` to specify how to include
  out-of-vocabulary values.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example with `num_oov_buckets`:
  File '/us/states.txt' contains 50 lines, each with a 2-character U.S. state
  abbreviation. All inputs with values in that file are assigned an ID 0-49,
  corresponding to its line number. All other values are hashed and assigned an
  ID 50-54.

  ```python
  import tensorflow as tf
  states = tf.feature_column.categorical_column_with_vocabulary_file(
    key='states', vocabulary_file='states.txt', vocabulary_size=50,
    num_oov_buckets=5)
  columns = [states]
  features = {'states':tf.constant([['california', 'georgia', 'michigan',
  'texas', 'new york'], ['new york', 'georgia', 'california', 'michigan',
  'texas']])}
  linear_prediction = tf.compat.v1.feature_column.linear_model(features,
  columns)
  ```

  Example with `default_value`:
  File '/us/states.txt' contains 51 lines - the first line is 'XX', and the
  other 50 each have a 2-character U.S. state abbreviation. Both a literal 'XX'
  in input, and other values missing from the file, will be assigned ID 0. All
  others are assigned the corresponding line number 1-50.

  ```python
  import tensorflow as tf
  states = tf.feature_column.categorical_column_with_vocabulary_file(
    key='states', vocabulary_file='states.txt', vocabulary_size=51,
    default_value=0)
  columns = [states]
  features = {'states':tf.constant([['california', 'georgia', 'michigan',
  'texas', 'new york'], ['new york', 'georgia', 'california', 'michigan',
  'texas']])}
  linear_prediction = tf.compat.v1.feature_column.linear_model(features,
  columns)
  ```

  And to make an embedding with either:

  ```python
  import tensorflow as tf
  states = tf.feature_column.categorical_column_with_vocabulary_file(
    key='states', vocabulary_file='states.txt', vocabulary_size=5,
    num_oov_buckets=1)
  columns = [tf.feature_column.embedding_column(states, 3)]
  features = {'states':tf.constant([['california', 'georgia', 'michigan',
  'texas', 'new york'], ['new york', 'georgia', 'california', 'michigan',
  'texas']])}
  input_layer = tf.keras.layers.DenseFeatures(columns)
  dense_tensor = input_layer(features)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the column
      name and the dictionary key for feature parsing configs, feature `Tensor`
      objects, and feature columns.
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
    A `CategoricalColumn` with a vocabulary file.

  Raises:
    ValueError: `vocabulary_file` is missing or cannot be opened.
    ValueError: `vocabulary_size` is missing or < 1.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: `dtype` is neither string nor integer.
  """
  return categorical_column_with_vocabulary_file_v2(key, vocabulary_file,
                                                    vocabulary_size, dtype,
                                                    default_value,
                                                    num_oov_buckets)


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export(
    'feature_column.categorical_column_with_vocabulary_file',
    v1=[])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def categorical_column_with_vocabulary_file_v2(key,
                                               vocabulary_file,
                                               vocabulary_size=None,
                                               dtype=dtypes.string,
                                               default_value=None,
                                               num_oov_buckets=0,
                                               file_format=None):
  # [FUNCTION DOC]
  # Public API (exported without the v2 suffix, deprecated at runtime) building a categorical column
  # whose ids are line numbers in a vocabulary file. Inputs: key, vocabulary file path, optional
  # vocabulary size, dtype, optional default id for unknown values, number of out-of-vocabulary
  # buckets, and optional file format (text, or gzip-compressed record file; inferred from a
  # file-name suffix when absent). When the size is absent the file must exist; the size is counted
  # from the file (via a dataset reduction for record files, converted to a plain number in eager
  # mode, else a graph tensor) and an informational log message is written. Raises value errors for
  # a missing file name, a nonexistent file, a size below 1 (non-tensor sizes only), both oov buckets
  # and a default value, a negative bucket count, or a dtype that is not string or integer. Returns a
  # VocabularyFileCategoricalColumn (default id -1 and bucket count 0 when not given).
  """A `CategoricalColumn` with a vocabulary file.

  Use this when your inputs are in string or integer format, and you have a
  vocabulary file that maps each value to an integer ID. By default,
  out-of-vocabulary values are ignored. Use either (but not both) of
  `num_oov_buckets` and `default_value` to specify how to include
  out-of-vocabulary values.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example with `num_oov_buckets`:
  File `'/us/states.txt'` contains 50 lines, each with a 2-character U.S. state
  abbreviation. All inputs with values in that file are assigned an ID 0-49,
  corresponding to its line number. All other values are hashed and assigned an
  ID 50-54.

  ```python
  states = categorical_column_with_vocabulary_file(
      key='states', vocabulary_file='/us/states.txt', vocabulary_size=50,
      num_oov_buckets=5)
  columns = [states, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  Example with `default_value`:
  File `'/us/states.txt'` contains 51 lines - the first line is `'XX'`, and the
  other 50 each have a 2-character U.S. state abbreviation. Both a literal
  `'XX'` in input, and other values missing from the file, will be assigned
  ID 0. All others are assigned the corresponding line number 1-50.

  ```python
  states = categorical_column_with_vocabulary_file(
      key='states', vocabulary_file='/us/states.txt', vocabulary_size=51,
      default_value=0)
  columns = [states, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  And to make an embedding with either:

  ```python
  columns = [embedding_column(states, 3),...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the column
      name and the dictionary key for feature parsing configs, feature `Tensor`
      objects, and feature columns.
    vocabulary_file: The vocabulary file name.
    vocabulary_size: Number of the elements in the vocabulary. This must be no
      greater than length of `vocabulary_file`, if less than length, later
      values are ignored. If None, it is set to the length of `vocabulary_file`.
    dtype: The type of features. Only string and integer types are supported.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of
      the input value. A positive `num_oov_buckets` can not be specified with
      `default_value`.
    file_format: The format of the vocabulary file. The format is 'text' by
      default unless `vocabulary_file` is a string which ends in 'tfrecord.gz'.
      Accepted alternative value for `file_format` is 'tfrecord_gzip'.

  Returns:
    A `CategoricalColumn` with a vocabulary file.

  Raises:
    ValueError: `vocabulary_file` is missing or cannot be opened.
    ValueError: `vocabulary_size` is missing or < 1.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: `dtype` is neither string nor integer.
  """
  if not vocabulary_file:
    raise ValueError('Missing vocabulary_file in {}.'.format(key))

  if file_format is None and vocabulary_file.endswith('tfrecord.gz'):
    file_format = 'tfrecord_gzip'

  if vocabulary_size is None:
    if not gfile.Exists(vocabulary_file):
      raise ValueError('vocabulary_file in {} does not exist.'.format(key))

    if file_format == 'tfrecord_gzip':
      ds = readers.TFRecordDataset(vocabulary_file, 'GZIP')
      vocabulary_size = ds.reduce(0, lambda x, _: x + 1)
      if context.executing_eagerly():
        vocabulary_size = vocabulary_size.numpy()
    else:
      with gfile.GFile(vocabulary_file, mode='rb') as f:
        vocabulary_size = sum(1 for _ in f)
    logging.info(
        'vocabulary_size = %d in %s is inferred from the number of elements '
        'in the vocabulary_file %s.', vocabulary_size, key, vocabulary_file)

  # `vocabulary_size` isn't required for lookup, but it is for `_num_buckets`.
  if not isinstance(vocabulary_size, tensor_lib.Tensor) and vocabulary_size < 1:
    raise ValueError('Invalid vocabulary_size in {}.'.format(key))
  if num_oov_buckets:
    if default_value is not None:
      raise ValueError(
          'Can\'t specify both num_oov_buckets and default_value in {}.'.format(
              key))
    if num_oov_buckets < 0:
      raise ValueError('Invalid num_oov_buckets {} in {}.'.format(
          num_oov_buckets, key))
  fc_utils.assert_string_or_int(dtype, prefix='column_name: {}'.format(key))
  fc_utils.assert_key_is_string(key)
  return VocabularyFileCategoricalColumn(
      key=key,
      vocabulary_file=vocabulary_file,
      vocabulary_size=vocabulary_size,
      num_oov_buckets=0 if num_oov_buckets is None else num_oov_buckets,
      default_value=-1 if default_value is None else default_value,
      dtype=dtype,
      file_format=file_format)


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.categorical_column_with_vocabulary_list')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def categorical_column_with_vocabulary_list(key,
                                            vocabulary_list,
                                            dtype=None,
                                            default_value=-1,
                                            num_oov_buckets=0):
  # [FUNCTION DOC]
  # Public API (deprecated at runtime) for a categorical column with an in-memory vocabulary; ids are
  # indices into the ordered list. Inputs: key, ordered vocabulary list, optional dtype (inferred from
  # the list when absent), default id for unknown values (default -1) and number of out-of-vocabulary
  # hash buckets (default zero). Raises value errors for an empty list, duplicate entries, both oov
  # buckets and a non -1 default value, negative bucket count, a dtype whose integer-ness differs
  # from the vocabulary, or a non string/integer dtype. Returns a VocabularyListCategoricalColumn
  # holding the list as a tuple.
  """A `CategoricalColumn` with in-memory vocabulary.

  Use this when your inputs are in string or integer format, and you have an
  in-memory vocabulary mapping each value to an integer ID. By default,
  out-of-vocabulary values are ignored. Use either (but not both) of
  `num_oov_buckets` and `default_value` to specify how to include
  out-of-vocabulary values.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example with `num_oov_buckets`:
  In the following example, each input in `vocabulary_list` is assigned an ID
  0-3 corresponding to its index (e.g., input 'B' produces output 2). All other
  inputs are hashed and assigned an ID 4-5.

  ```python
  colors = categorical_column_with_vocabulary_list(
      key='colors', vocabulary_list=('R', 'G', 'B', 'Y'),
      num_oov_buckets=2)
  columns = [colors, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  Example with `default_value`:
  In the following example, each input in `vocabulary_list` is assigned an ID
  0-4 corresponding to its index (e.g., input 'B' produces output 3). All other
  inputs are assigned `default_value` 0.


  ```python
  colors = categorical_column_with_vocabulary_list(
      key='colors', vocabulary_list=('X', 'R', 'G', 'B', 'Y'), default_value=0)
  columns = [colors, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  And to make an embedding with either:

  ```python
  columns = [embedding_column(colors, 3),...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the column
      name and the dictionary key for feature parsing configs, feature `Tensor`
      objects, and feature columns.
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
    A `CategoricalColumn` with in-memory vocabulary.

  Raises:
    ValueError: if `vocabulary_list` is empty, or contains duplicate keys.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: if `dtype` is not integer or string.
  """
  if (vocabulary_list is None) or (len(vocabulary_list) < 1):
    raise ValueError(
        'vocabulary_list {} must be non-empty, column_name: {}'.format(
            vocabulary_list, key))
  if len(set(vocabulary_list)) != len(vocabulary_list):
    raise ValueError(
        'Duplicate keys in vocabulary_list {}, column_name: {}'.format(
            vocabulary_list, key))
  vocabulary_dtype = dtypes.as_dtype(np.array(vocabulary_list).dtype)
  if num_oov_buckets:
    if default_value != -1:
      raise ValueError(
          'Can\'t specify both num_oov_buckets and default_value in {}.'.format(
              key))
    if num_oov_buckets < 0:
      raise ValueError('Invalid num_oov_buckets {} in {}.'.format(
          num_oov_buckets, key))
  fc_utils.assert_string_or_int(
      vocabulary_dtype, prefix='column_name: {} vocabulary'.format(key))
  if dtype is None:
    dtype = vocabulary_dtype
  elif dtype.is_integer != vocabulary_dtype.is_integer:
    raise ValueError(
        'dtype {} and vocabulary dtype {} do not match, column_name: {}'.format(
            dtype, vocabulary_dtype, key))
  fc_utils.assert_string_or_int(dtype, prefix='column_name: {}'.format(key))
  fc_utils.assert_key_is_string(key)

  return VocabularyListCategoricalColumn(
      key=key,
      vocabulary_list=tuple(vocabulary_list),
      dtype=dtype,
      default_value=default_value,
      num_oov_buckets=num_oov_buckets)


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.categorical_column_with_identity')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def categorical_column_with_identity(key, num_buckets, default_value=None):
  # [FUNCTION DOC]
  # Public API (deprecated at runtime) for a categorical column whose id is the integer input value
  # itself, in the range zero to the bucket count. Inputs: string key, number of buckets (at least 1)
  # and optional default id used for out-of-range values. Raises value errors when the bucket count
  # is below 1 or the default is outside the valid range, and an error for a non-string key. Returns
  # an IdentityCategoricalColumn.
  """A `CategoricalColumn` that returns identity values.

  Use this when your inputs are integers in the range `[0, num_buckets)`, and
  you want to use the input value itself as the categorical ID. Values outside
  this range will result in `default_value` if specified, otherwise it will
  fail.

  Typically, this is used for contiguous ranges of integer indexes, but
  it doesn't have to be. This might be inefficient, however, if many of IDs
  are unused. Consider `categorical_column_with_hash_bucket` in that case.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  In the following examples, each input in the range `[0, 1000000)` is assigned
  the same value. All other inputs are assigned `default_value` 0. Note that a
  literal 0 in inputs will result in the same default ID.

  Linear model:

  ```python
  import tensorflow as tf
  video_id = tf.feature_column.categorical_column_with_identity(
      key='video_id', num_buckets=1000000, default_value=0)
  columns = [video_id]
  features = {'video_id': tf.sparse.from_dense([[2, 85, 0, 0, 0],
  [33,78, 2, 73, 1]])}
  linear_prediction = tf.compat.v1.feature_column.linear_model(features,
  columns)
  ```

  Embedding for a DNN model:

  ```python
  import tensorflow as tf
  video_id = tf.feature_column.categorical_column_with_identity(
      key='video_id', num_buckets=1000000, default_value=0)
  columns = [tf.feature_column.embedding_column(video_id, 9)]
  features = {'video_id': tf.sparse.from_dense([[2, 85, 0, 0, 0],
  [33,78, 2, 73, 1]])}
  input_layer = tf.keras.layers.DenseFeatures(columns)
  dense_tensor = input_layer(features)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the column
      name and the dictionary key for feature parsing configs, feature `Tensor`
      objects, and feature columns.
    num_buckets: Range of inputs and outputs is `[0, num_buckets)`.
    default_value: If set, values outside of range `[0, num_buckets)` will be
      replaced with this value. If not set, values >= num_buckets will cause a
      failure while values < 0 will be dropped.

  Returns:
    A `CategoricalColumn` that returns identity values.

  Raises:
    ValueError: if `num_buckets` is less than one.
    ValueError: if `default_value` is not in range `[0, num_buckets)`.
  """
  if num_buckets < 1:
    raise ValueError('num_buckets {} < 1, column_name {}'.format(
        num_buckets, key))
  if (default_value is not None) and ((default_value < 0) or
                                      (default_value >= num_buckets)):
    raise ValueError(
        'default_value {} not in range [0, {}), column_name {}'.format(
            default_value, num_buckets, key))
  fc_utils.assert_key_is_string(key)
  return IdentityCategoricalColumn(
      key=key, number_buckets=num_buckets, default_value=default_value)


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.indicator_column')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def indicator_column(categorical_column):
  # [FUNCTION DOC]
  # Public API (deprecated at runtime) that wraps a categorical column into a multi-hot dense column
  # (a count vector over its buckets). Input: a categorical column (v1 or v2). Raises a value error if
  # it is not a categorical column. Returns an IndicatorColumn; no other side effects.
  """Represents multi-hot representation of given categorical column.

  - For DNN model, `indicator_column` can be used to wrap any
    `categorical_column_*` (e.g., to feed to DNN). Consider to Use
    `embedding_column` if the number of buckets/unique(values) are large.

  - For Wide (aka linear) model, `indicator_column` is the internal
    representation for categorical column when passing categorical column
    directly (as any element in feature_columns) to `linear_model`. See
    `linear_model` for details.

  ```python
  name = indicator_column(categorical_column_with_vocabulary_list(
      'name', ['bob', 'george', 'wanda']))
  columns = [name, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)

  dense_tensor == [[1, 0, 0]]  # If "name" bytes_list is ["bob"]
  dense_tensor == [[1, 0, 1]]  # If "name" bytes_list is ["bob", "wanda"]
  dense_tensor == [[2, 0, 0]]  # If "name" bytes_list is ["bob", "bob"]
  ```

  Args:
    categorical_column: A `CategoricalColumn` which is created by
      `categorical_column_with_*` or `crossed_column` functions.

  Returns:
    An `IndicatorColumn`.

  Raises:
    ValueError: If `categorical_column` is not CategoricalColumn type.
  """
  if not isinstance(categorical_column,
                    (CategoricalColumn, fc_old._CategoricalColumn)):  # pylint: disable=protected-access
    raise ValueError(
        'Unsupported input type. Input must be a CategoricalColumn. '
        'Given: {}'.format(categorical_column))
  return IndicatorColumn(categorical_column)


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.weighted_categorical_column')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def weighted_categorical_column(categorical_column,
                                weight_feature_key,
                                dtype=dtypes.float32):
  # [FUNCTION DOC]
  # Public API (deprecated at runtime) pairing a categorical column with a second sparse feature that
  # supplies per-id weights. Inputs: the categorical column, the key of the weight feature, and the
  # weight dtype (default float32; must be integer or floating). Raises a value error when the dtype
  # is missing or not numeric. Returns a WeightedCategoricalColumn.
  """Applies weight values to a `CategoricalColumn`.

  Use this when each of your sparse inputs has both an ID and a value. For
  example, if you're representing text documents as a collection of word
  frequencies, you can provide 2 parallel sparse input features ('terms' and
  'frequencies' below).

  Example:

  Input `tf.Example` objects:

  ```proto
  [
    features {
      feature {
        key: "terms"
        value {bytes_list {value: "very" value: "model"}}
      }
      feature {
        key: "frequencies"
        value {float_list {value: 0.3 value: 0.1}}
      }
    },
    features {
      feature {
        key: "terms"
        value {bytes_list {value: "when" value: "course" value: "human"}}
      }
      feature {
        key: "frequencies"
        value {float_list {value: 0.4 value: 0.1 value: 0.2}}
      }
    }
  ]
  ```

  ```python
  categorical_column = categorical_column_with_hash_bucket(
      column_name='terms', hash_bucket_size=1000)
  weighted_column = weighted_categorical_column(
      categorical_column=categorical_column, weight_feature_key='frequencies')
  columns = [weighted_column, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  This assumes the input dictionary contains a `SparseTensor` for key
  'terms', and a `SparseTensor` for key 'frequencies'. These 2 tensors must have
  the same indices and dense shape.

  Args:
    categorical_column: A `CategoricalColumn` created by
      `categorical_column_with_*` functions.
    weight_feature_key: String key for weight values.
    dtype: Type of weights, such as `tf.float32`. Only float and integer weights
      are supported.

  Returns:
    A `CategoricalColumn` composed of two sparse features: one represents id,
    the other represents weight (value) of the id feature in that example.

  Raises:
    ValueError: if `dtype` is not convertible to float.
  """
  if (dtype is None) or not (dtype.is_integer or dtype.is_floating):
    raise ValueError('dtype {} is not convertible to float.'.format(dtype))
  return WeightedCategoricalColumn(
      categorical_column=categorical_column,
      weight_feature_key=weight_feature_key,
      dtype=dtype)


@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.crossed_column')
@deprecation.deprecated(
    None,
    'Use `tf.keras.layers.experimental.preprocessing.HashedCrossing` '
    'instead for feature crossing when preprocessing data to train a '
    'Keras model.')
def crossed_column(keys, hash_bucket_size, hash_key=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: Docs and the error text require a bucket size greater than 1, but the check only
  # rejects values below 1, so a size of 1 is accepted.
  # Public API (deprecated at runtime with advice to use hashed crossing) that builds a hashed
  # feature cross. Inputs: an iterable of keys (strings or categorical columns, at least two, hashed
  # categorical columns not allowed), the number of hash buckets, and an optional hash key for the
  # fingerprint combiner. Raises value errors for a missing or too small bucket count, fewer than two
  # keys, an unsupported key type, or a hashed categorical key. Returns a CrossedColumn with the keys
  # stored as a tuple.
  """Returns a column for performing crosses of categorical features.

  Crossed features will be hashed according to `hash_bucket_size`. Conceptually,
  the transformation can be thought of as:
    Hash(cartesian product of features) % `hash_bucket_size`

  For example, if the input features are:

  * SparseTensor referred by first key:

    ```python
    shape = [2, 2]
    {
        [0, 0]: "a"
        [1, 0]: "b"
        [1, 1]: "c"
    }
    ```

  * SparseTensor referred by second key:

    ```python
    shape = [2, 1]
    {
        [0, 0]: "d"
        [1, 0]: "e"
    }
    ```

  then crossed feature will look like:

  ```python
   shape = [2, 2]
  {
      [0, 0]: Hash64("d", Hash64("a")) % hash_bucket_size
      [1, 0]: Hash64("e", Hash64("b")) % hash_bucket_size
      [1, 1]: Hash64("e", Hash64("c")) % hash_bucket_size
  }
  ```

  Here is an example to create a linear model with crosses of string features:

  ```python
  keywords_x_doc_terms = crossed_column(['keywords', 'doc_terms'], 50K)
  columns = [keywords_x_doc_terms, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  You could also use vocabulary lookup before crossing:

  ```python
  keywords = categorical_column_with_vocabulary_file(
      'keywords', '/path/to/vocabulary/file', vocabulary_size=1K)
  keywords_x_doc_terms = crossed_column([keywords, 'doc_terms'], 50K)
  columns = [keywords_x_doc_terms, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  If an input feature is of numeric type, you can use
  `categorical_column_with_identity`, or `bucketized_column`, as in the example:

  ```python
  # vertical_id is an integer categorical feature.
  vertical_id = categorical_column_with_identity('vertical_id', 10K)
  price = numeric_column('price')
  # bucketized_column converts numerical feature to a categorical one.
  bucketized_price = bucketized_column(price, boundaries=[...])
  vertical_id_x_price = crossed_column([vertical_id, bucketized_price], 50K)
  columns = [vertical_id_x_price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  To use crossed column in DNN model, you need to add it in an embedding column
  as in this example:

  ```python
  vertical_id_x_price = crossed_column([vertical_id, bucketized_price], 50K)
  vertical_id_x_price_embedded = embedding_column(vertical_id_x_price, 10)
  dense_tensor = input_layer(features, [vertical_id_x_price_embedded, ...])
  ```

  Args:
    keys: An iterable identifying the features to be crossed. Each element can
      be either:
      * string: Will use the corresponding feature which must be of string type.
      * `CategoricalColumn`: Will use the transformed tensor produced by this
        column. Does not support hashed categorical column.
    hash_bucket_size: An int > 1. The number of buckets.
    hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
      function to combine the crosses fingerprints on SparseCrossOp (optional).

  Returns:
    A `CrossedColumn`.

  Raises:
    ValueError: If `len(keys) < 2`.
    ValueError: If any of the keys is neither a string nor `CategoricalColumn`.
    ValueError: If any of the keys is `HashedCategoricalColumn`.
    ValueError: If `hash_bucket_size < 1`.
  """
  if not hash_bucket_size or hash_bucket_size < 1:
    raise ValueError('hash_bucket_size must be > 1. '
                     'hash_bucket_size: {}'.format(hash_bucket_size))
  if not keys or len(keys) < 2:
    raise ValueError(
        'keys must be a list with length > 1. Given: {}'.format(keys))
  for key in keys:
    if (not isinstance(key, six.string_types) and
        not isinstance(key, (CategoricalColumn, fc_old._CategoricalColumn))):  # pylint: disable=protected-access
      raise ValueError(
          'Unsupported key type. All keys must be either string, or '
          'categorical column except HashedCategoricalColumn. '
          'Given: {}'.format(key))
    if isinstance(key,
                  (HashedCategoricalColumn, fc_old._HashedCategoricalColumn)):  # pylint: disable=protected-access
      raise ValueError(
          'categorical_column_with_hash_bucket is not supported for crossing. '
          'Hashing before crossing will increase probability of collision. '
          'Instead, use the feature name as a string. Given: {}'.format(key))
  return CrossedColumn(
      keys=tuple(keys), hash_bucket_size=hash_bucket_size, hash_key=hash_key)


# TODO(b/181853833): Add a tf.type for instance type checking.
@tf_export('__internal__.feature_column.DenseColumn', v1=[])
class DenseColumn(fc_types.FeatureColumn):
  # [CLASS DOC]
  # Abstract base class (exported as an internal symbol) for columns that can produce a dense tensor,
  # such as numeric, embedding and indicator columns. Declares an abstract variable_shape property and
  # an abstract method to build the dense tensor; subclasses supply both.
  """Represents a column which can be represented as `Tensor`.

  Some examples of this type are: numeric_column, embedding_column,
  indicator_column.
  """

  @abc.abstractproperty
  def variable_shape(self):
    # [FUNCTION DOC]
    # Abstract property: shape of the dense tensor this column produces, excluding the batch
    # dimension. The base implementation does nothing (returns nothing); subclasses return a tensor shape.
    """`TensorShape` of `get_dense_tensor`, without batch dimension."""
    pass

  @abc.abstractmethod
  def get_dense_tensor(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Abstract: builds the dense tensor for this column. Inputs: a transformation cache giving access to
    # the raw and transformed features, and a state manager for variables and lookup tables. Expected
    # result in subclasses: a tensor of shape batch size followed by variable_shape. The base
    # implementation does nothing and returns nothing.
    """Returns a `Tensor`.

    The output of this function will be used by model-builder-functions. For
    example the pseudo code of `input_layer` will be like:

    ```python
    def input_layer(features, feature_columns, ...):
      outputs = [fc.get_dense_tensor(...) for fc in feature_columns]
      return tf.concat(outputs)
    ```

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      `Tensor` of shape [batch_size] + `variable_shape`.
    """
    pass


def is_feature_column_v2(feature_columns):
  # [FUNCTION DOC]
  # Returns true only if every item in the given iterable is a FeatureColumn instance whose internal
  # v2 flag is set; returns false at the first item that is not or is not v2. An empty input yields
  # true. Pure check with no side effects.
  """Returns True if all feature columns are V2."""
  for feature_column in feature_columns:
    if not isinstance(feature_column, fc_types.FeatureColumn):
      return False
    if not feature_column._is_v2_column:  # pylint: disable=protected-access
      return False
  return True


def _create_weighted_sum(column, transformation_cache, state_manager,
                         sparse_combiner, weight_var):
  # [FUNCTION DOC]
  # HELPER: Dispatcher used when building the linear model's weighted sums.
  # Inputs: a column, a transformation cache, a state manager, a combiner name for sparse columns and
  # the weight variable. Delegates to the categorical weighted-sum helper when the column is a
  # categorical column, otherwise to the dense one (the sparse combiner is unused in that case).
  # Returns the resulting weighted-sum tensor.
  """Creates a weighted sum for a dense/categorical column for linear_model."""
  if isinstance(column, CategoricalColumn):
    return _create_categorical_column_weighted_sum(
        column=column,
        transformation_cache=transformation_cache,
        state_manager=state_manager,
        sparse_combiner=sparse_combiner,
        weight_var=weight_var)
  else:
    return _create_dense_column_weighted_sum(
        column=column,
        transformation_cache=transformation_cache,
        state_manager=state_manager,
        weight_var=weight_var)


def _create_dense_column_weighted_sum(column, transformation_cache,
                                      state_manager, weight_var):
  # [FUNCTION DOC]
  # HELPER: Builds the linear-model contribution of a dense column.
  # Gets the dense tensor of the column, flattens everything after the batch dimension into one axis
  # of the column's element count, and multiplies with the given weight variable (op named weighted
  # sum). Returns the matrix product tensor.
  """Create a weighted sum of a dense column for linear_model."""
  tensor = column.get_dense_tensor(transformation_cache, state_manager)
  num_elements = column.variable_shape.num_elements()
  batch_size = array_ops.shape(tensor)[0]
  tensor = array_ops.reshape(tensor, shape=(batch_size, num_elements))
  return math_ops.matmul(tensor, weight_var, name='weighted_sum')


class CategoricalColumn(fc_types.FeatureColumn):
  # [CLASS DOC]
  # Abstract base class for columns that represent categorical data as sparse id tensors, optionally
  # with weights. Defines a named pair type (IdWeightPair with an id tensor and a weight tensor) used
  # as the return type of get_sparse_tensors, an abstract bucket count property and an abstract method
  # to produce the pair. Concrete columns (hashed, vocabulary, identity, weighted, crossed,
  # bucketized, sequence) subclass it.
  """Represents a categorical feature.

  A categorical feature typically handled with a `tf.sparse.SparseTensor` of
  IDs.
  """

  IdWeightPair = collections.namedtuple(  # pylint: disable=invalid-name
      'IdWeightPair', ('id_tensor', 'weight_tensor'))

  @abc.abstractproperty
  def num_buckets(self):
    # [FUNCTION DOC]
    # Abstract property: number of distinct id buckets of this sparse feature. Base implementation
    # does nothing and returns nothing; subclasses return an integer.
    """Returns number of buckets in this sparse feature."""
    pass

  @abc.abstractmethod
  def get_sparse_tensors(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Abstract: produces an IdWeightPair of sparse tensors (ids and optional weights, where nothing
    # for weights means all weights are one). Inputs: a transformation cache and a state manager.
    # The base implementation does nothing and returns nothing; subclasses return the pair.
    """Returns an IdWeightPair.

    `IdWeightPair` is a pair of `SparseTensor`s which represents ids and
    weights.

    `IdWeightPair.id_tensor` is typically a `batch_size` x `num_buckets`
    `SparseTensor` of `int64`. `IdWeightPair.weight_tensor` is either a
    `SparseTensor` of `float` or `None` to indicate all weights should be
    taken to be 1. If specified, `weight_tensor` must have exactly the same
    shape and indices as `sp_ids`. Expected `SparseTensor` is same as parsing
    output of a `VarLenFeature` which is a ragged matrix.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.
    """
    pass


def _create_categorical_column_weighted_sum(column, transformation_cache,
                                            state_manager, sparse_combiner,
                                            weight_var):
  # [FUNCTION DOC]
  # HELPER: Builds the linear-model contribution of a categorical column.
  # Gets the column's id and weight sparse tensors, reshapes both to two dimensions (batch size by
  # unknown), and applies a safe sparse embedding lookup on the weight variable with the given
  # combiner (named weighted sum), which equals the sum of weights selected by the multi-hot input.
  # Inputs: column, transformation cache, state manager, combiner name, weight variable. Returns the
  # looked-up tensor.
  # pylint: disable=g-doc-return-or-yield,g-doc-args
  """Create a weighted sum of a categorical column for linear_model.

  Note to maintainer: As implementation details, the weighted sum is
  implemented via embedding_lookup_sparse toward efficiency. Mathematically,
  they are the same.

  To be specific, conceptually, categorical column can be treated as multi-hot
  vector. Say:

  ```python
    x = [0 0 1]  # categorical column input
    w = [a b c]  # weights
  ```
  The weighted sum is `c` in this case, which is same as `w[2]`.

  Another example is

  ```python
    x = [0 1 1]  # categorical column input
    w = [a b c]  # weights
  ```
  The weighted sum is `b + c` in this case, which is same as `w[2] + w[3]`.

  For both cases, we can implement weighted sum via embedding_lookup with
  sparse_combiner = "sum".
  """

  sparse_tensors = column.get_sparse_tensors(transformation_cache,
                                             state_manager)
  id_tensor = sparse_ops.sparse_reshape(
      sparse_tensors.id_tensor,
      [array_ops.shape(sparse_tensors.id_tensor)[0], -1])
  weight_tensor = sparse_tensors.weight_tensor
  if weight_tensor is not None:
    weight_tensor = sparse_ops.sparse_reshape(
        weight_tensor, [array_ops.shape(weight_tensor)[0], -1])

  return embedding_ops.safe_embedding_lookup_sparse(
      weight_var,
      id_tensor,
      sparse_weights=weight_tensor,
      combiner=sparse_combiner,
      name='weighted_sum')


# TODO(b/181853833): Add a tf.type for instance type checking.
@tf_export('__internal__.feature_column.SequenceDenseColumn', v1=[])
@serialization.register_feature_column
class SequenceDenseColumn(fc_types.FeatureColumn):
  # [CLASS DOC]
  # Abstract base class (exported as an internal symbol and registered with the serialization
  # registry) for dense columns that represent sequences. Defines a named pair type holding a dense
  # tensor and the per-example sequence lengths, and one abstract method to obtain that pair.
  """Represents dense sequence data."""

  TensorSequenceLengthPair = collections.namedtuple(  # pylint: disable=invalid-name
      'TensorSequenceLengthPair', ('dense_tensor', 'sequence_length'))

  @abc.abstractmethod
  def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Abstract: returns the dense tensor and sequence length pair for this column. Inputs: a
    # transformation cache and a state manager. Base implementation does nothing and returns nothing.
    """Returns a `TensorSequenceLengthPair`.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.
    """
    pass


@tf_export('__internal__.feature_column.FeatureTransformationCache', v1=[])
class FeatureTransformationCache(object):
  # [CLASS DOC]
  # Memoising front end over a dictionary of raw features used while a model is built (exported as
  # an internal symbol). Keys are either plain strings (raw features, converted to tensors on first
  # use, rank-one inputs expanded to rank two) or feature columns (whose transformed output is
  # computed once by the column and reused). Prevents duplicated operations when a feature is used by
  # several columns. Holds a private copy of the feature mapping and a cache of resulting tensors.
  """Handles caching of transformations while building the model.

  `FeatureColumn` specifies how to digest an input column to the network. Some
  feature columns require data transformations. This class caches those
  transformations.

  Some features may be used in more than one place. For example, one can use a
  bucketized feature by itself and a cross with it. In that case we
  should create only one bucketization op instead of creating ops for each
  feature column separately. To handle re-use of transformed columns,
  `FeatureTransformationCache` caches all previously transformed columns.

  Example:
  We're trying to use the following `FeatureColumn`s:

  ```python
  bucketized_age = fc.bucketized_column(fc.numeric_column("age"), ...)
  keywords = fc.categorical_column_with_hash_buckets("keywords", ...)
  age_X_keywords = fc.crossed_column([bucketized_age, "keywords"])
  ... = linear_model(features,
                          [bucketized_age, keywords, age_X_keywords]
  ```

  If we transform each column independently, then we'll get duplication of
  bucketization (one for cross, one for bucketization itself).
  The `FeatureTransformationCache` eliminates this duplication.
  """

  def __init__(self, features):
    # [FUNCTION DOC]
    # Creates the cache. Input: a mapping from feature name (string) or feature column to tensors or
    # sparse tensors (or values convertible to them); a shallow copy is stored so later changes by the
    # caller do not affect the cache. Also initialises an empty cache of produced tensors. Returns
    # nothing.
    """Creates a `FeatureTransformationCache`.

    Args:
      features: A mapping from feature column to objects that are `Tensor` or
        `SparseTensor`, or can be converted to same via
        `sparse_tensor.convert_to_tensor_or_sparse_tensor`. A `string` key
        signifies a base feature (not-transformed). A `FeatureColumn` key means
        that this `Tensor` is the output of an existing `FeatureColumn` which
        can be reused.
    """
    self._features = features.copy()
    self._feature_tensors = {}

  def get(self, key, state_manager, training=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring lists only a value error, but a type error is also raised when the key
    # is neither a string nor a feature column. In addition, a type error raised from inside a column's
    # own transformation is swallowed by the retry without the training argument.
    # Returns the (cached) tensor for a key. Inputs: a string or feature column; a state manager; an
    # optional training flag forwarded to columns whose transform method accepts it. Order: cached
    # result, then raw feature (converted, cached, returned), then error for unknown string (value
    # error), type error for other non-column keys; otherwise logs a debug message and asks the
    # column to transform, first with the training flag and, if that raises a type error, again without
    # it. Raises a value error when the column returns nothing. Stores and returns the result.
    """Returns a `Tensor` for the given key.

    A `str` key is used to access a base feature (not-transformed). When a
    `FeatureColumn` is passed, the transformed feature is returned if it
    already exists, otherwise the given `FeatureColumn` is asked to provide its
    transformed output, which is then cached.

    Args:
      key: a `str` or a `FeatureColumn`.
      state_manager: A StateManager object that holds the FeatureColumn state.
      training: Boolean indicating whether to the column is being used in
        training mode. This argument is passed to the transform_feature method
        of any `FeatureColumn` that takes a `training` argument. For example, if
        a `FeatureColumn` performed dropout, it could expose a `training`
        argument to control whether the dropout should be applied.

    Returns:
      The transformed `Tensor` corresponding to the `key`.

    Raises:
      ValueError: if key is not found or a transformed `Tensor` cannot be
        computed.
    """
    if key in self._feature_tensors:
      # FeatureColumn is already transformed or converted.
      return self._feature_tensors[key]

    if key in self._features:
      feature_tensor = self._get_raw_feature_as_tensor(key)
      self._feature_tensors[key] = feature_tensor
      return feature_tensor

    if isinstance(key, six.string_types):
      raise ValueError('Feature {} is not in features dictionary.'.format(key))

    if not isinstance(key, fc_types.FeatureColumn):
      raise TypeError('"key" must be either a "str" or "FeatureColumn". '
                      'Provided: {}'.format(key))

    column = key
    logging.debug('Transforming feature_column %s.', column)

    # Some columns may need information about whether the transformation is
    # happening in training or prediction mode, but not all columns expose this
    # argument.
    try:
      transformed = column.transform_feature(
          self, state_manager, training=training)
    except TypeError:
      transformed = column.transform_feature(self, state_manager)
    if transformed is None:
      raise ValueError('Column {} is not supported.'.format(column.name))
    self._feature_tensors[column] = transformed
    return transformed

  def _get_raw_feature_as_tensor(self, key):
    # [FUNCTION DOC]
    # HELPER: Converts a raw feature to a tensor for the cache.
    # Input: a string key present in the feature mapping. Converts the value to a tensor or sparse
    # tensor and makes it at least rank two by adding a trailing axis when the rank is one (sparse
    # tensors are reshaped, dense ones expanded). Rank zero raises a value error when the rank is
    # statically known; with unknown rank an assertion is attached and the expansion is chosen at
    # run time by a conditional. Returns the tensor or sparse tensor.
    """Gets the raw_feature (keyed by `key`) as `tensor`.

    The raw feature is converted to (sparse) tensor and maybe expand dim.

    For both `Tensor` and `SparseTensor`, the rank will be expanded (to 2) if
    the rank is 1. This supports dynamic rank also. For rank 0 raw feature, will
    error out as it is not supported.

    Args:
      key: A `str` key to access the raw feature.

    Returns:
      A `Tensor` or `SparseTensor`.

    Raises:
      ValueError: if the raw feature has rank 0.
    """
    raw_feature = self._features[key]
    feature_tensor = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(
        raw_feature)

    def expand_dims(input_tensor):
      # [FUNCTION DOC]
      # HELPER: Nested utility for the raw feature conversion. Turns a rank-one tensor into rank two by
      # appending an axis of size one: a sparse reshape for sparse tensors, a dense expansion otherwise.
      # Input: the tensor; returns the expanded tensor. Assumes rank one without checking.
      # Input_tensor must have rank 1.
      if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
        return sparse_ops.sparse_reshape(input_tensor,
                                         [array_ops.shape(input_tensor)[0], 1])
      else:
        return array_ops.expand_dims(input_tensor, -1)

    rank = feature_tensor.get_shape().ndims
    if rank is not None:
      if rank == 0:
        raise ValueError(
            'Feature (key: {}) cannot have rank 0. Given: {}'.format(
                key, feature_tensor))
      return feature_tensor if rank != 1 else expand_dims(feature_tensor)

    # Handle dynamic rank.
    with ops.control_dependencies([
        check_ops.assert_positive(
            array_ops.rank(feature_tensor),
            message='Feature (key: {}) cannot have rank 0. Given: {}'.format(
                key, feature_tensor))
    ]):
      return cond.cond(
          math_ops.equal(1, array_ops.rank(feature_tensor)),
          lambda: expand_dims(feature_tensor), lambda: feature_tensor)


# TODO(ptucker): Move to third_party/tensorflow/python/ops/sparse_ops.py
def _to_sparse_input_and_drop_ignore_values(input_tensor, ignore_value=None):
  # [FUNCTION DOC]
  # HELPER: Used by the categorical columns to normalise their input into a sparse tensor.
  # KNOWN ISSUE: The docstring says a value error is raised when the input rank is unknown, but no
  # such check exists in the code.
  # Converts the input to a tensor or sparse tensor; sparse inputs are returned unchanged. Dense
  # inputs become sparse by dropping cells equal to an ignore value; when none is given it is the empty
  # string for string tensors, minus one for integer tensors, and the zero value of the numpy type
  # otherwise. Returns a sparse tensor with the original dense shape (64-bit integer indices).
  """Converts a `Tensor` to a `SparseTensor`, dropping ignore_value cells.

  If `input_tensor` is already a `SparseTensor`, just return it.

  Args:
    input_tensor: A string or integer `Tensor`.
    ignore_value: Entries in `dense_tensor` equal to this value will be absent
      from the resulting `SparseTensor`. If `None`, default value of
      `dense_tensor`'s dtype will be used ('' for `str`, -1 for `int`).

  Returns:
    A `SparseTensor` with the same shape as `input_tensor`.

  Raises:
    ValueError: when `input_tensor`'s rank is `None`.
  """
  input_tensor = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(
      input_tensor)
  if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
    return input_tensor
  with ops.name_scope(None, 'to_sparse_input', (
      input_tensor,
      ignore_value,
  )):
    if ignore_value is None:
      if input_tensor.dtype == dtypes.string:
        # Exception due to TF strings are converted to numpy objects by default.
        ignore_value = ''
      elif input_tensor.dtype.is_integer:
        ignore_value = -1  # -1 has a special meaning of missing feature
      else:
        # NOTE: `as_numpy_dtype` is a property, so with the parentheses this is
        # constructing a new numpy object of the given type, which yields the
        # default value for that type.
        ignore_value = input_tensor.dtype.as_numpy_dtype()
    ignore_value = math_ops.cast(
        ignore_value, input_tensor.dtype, name='ignore_value')
    indices = array_ops.where_v2(
        math_ops.not_equal(input_tensor, ignore_value), name='indices')
    return sparse_tensor_lib.SparseTensor(
        indices=indices,
        values=array_ops.gather_nd(input_tensor, indices, name='values'),
        dense_shape=array_ops.shape(
            input_tensor, out_type=dtypes.int64, name='dense_shape'))


def _normalize_feature_columns(feature_columns):
  # [FUNCTION DOC]
  # HELPER: Validates and canonicalises user supplied feature columns.
  # Input: a single feature column, an iterator or other iterable of columns. A single column becomes
  # a one-element list and an iterator is materialised. Raises value errors for a dictionary, items
  # that are not feature columns, an empty collection, or two columns sharing a name. Returns the
  # columns sorted by name.
  """Normalizes the `feature_columns` input.

  This method converts the `feature_columns` to list type as best as it can. In
  addition, verifies the type and other parts of feature_columns, required by
  downstream library.

  Args:
    feature_columns: The raw feature columns, usually passed by users.

  Returns:
    The normalized feature column list.

  Raises:
    ValueError: for any invalid inputs, such as empty, duplicated names, etc.
  """
  if isinstance(feature_columns, fc_types.FeatureColumn):
    feature_columns = [feature_columns]

  if isinstance(feature_columns, collections_abc.Iterator):
    feature_columns = list(feature_columns)

  if isinstance(feature_columns, dict):
    raise ValueError('Expected feature_columns to be iterable, found dict.')

  for column in feature_columns:
    if not isinstance(column, fc_types.FeatureColumn):
      raise ValueError('Items of feature_columns must be a FeatureColumn. '
                       'Given (type {}): {}.'.format(type(column), column))
  if not feature_columns:
    raise ValueError('feature_columns must not be empty.')
  name_to_column = {}
  for column in feature_columns:
    if column.name in name_to_column:
      raise ValueError('Duplicate feature column name found for columns: {} '
                       'and {}. This usually means that these columns refer to '
                       'same base feature. Either one must be discarded or a '
                       'duplicated but renamed item must be inserted in '
                       'features dict.'.format(column,
                                               name_to_column[column.name]))
    name_to_column[column.name] = column

  return sorted(feature_columns, key=lambda x: x.name)


@serialization.register_feature_column
class NumericColumn(
    DenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'NumericColumn',
        ('key', 'shape', 'default_value', 'dtype', 'normalizer_fn'))):
  # [CLASS DOC]
  # Immutable named tuple (key, shape, default_value, dtype, normalizer_fn) produced by the
  # numeric_column factory. Represents a dense real-valued feature parsed as fixed-length data and
  # optionally normalised; it is both a v2 DenseColumn and a legacy dense column. Registered for
  # serialization.
  """see `numeric_column`."""

  @property
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # Internal property marking the column as a v2 column; always returns true.
    return True

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property returning the column name, which is the feature key.
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # Property returning a one-entry mapping from the key to a fixed-length feature description built
    # from the column's shape, dtype and default value.
    """See `FeatureColumn` base class."""
    return {
        self.key:
            parsing_ops.FixedLenFeature(self.shape, self.dtype,
                                        self.default_value)
    }

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property; returns the same mapping as parse_example_spec.
    # Emits a deprecation warning when accessed.
    return self.parse_example_spec

  def _transform_input_tensor(self, input_tensor):
    # [FUNCTION DOC]
    # HELPER: Shared by the v1 and v2 transform paths. Input: the raw input tensor. Raises a value
    # error if it is a sparse tensor. Applies the normalizer callable when one is configured and casts
    # the result to 32-bit float. Returns the resulting tensor.
    if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
      raise ValueError(
          'The corresponding Tensor of numerical column must be a Tensor. '
          'SparseTensor is not supported. key: {}'.format(self.key))
    if self.normalizer_fn is not None:
      input_tensor = self.normalizer_fn(input_tensor)
    return math_ops.cast(input_tensor, dtypes.float32)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) transform. Input: the legacy input cache object. Fetches the tensor
    # stored under the column key and returns it normalised and cast to float via the shared helper.
    # Emits a deprecation warning.
    input_tensor = inputs.get(self.key)
    return self._transform_input_tensor(input_tensor)

  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Applies the column's transformation: reads the raw tensor for the key from the transformation
    # cache (using the state manager), then normalises it with the normalizer function if any and casts
    # to float. Inputs: transformation cache and state manager. Returns the normalised tensor. Raises a
    # value error if the raw feature is sparse.
    """See `FeatureColumn` base class.

    In this case, we apply the `normalizer_fn` to the input tensor.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Normalized input tensor.
    Raises:
      ValueError: If a SparseTensor is passed in.
    """
    input_tensor = transformation_cache.get(self.key, state_manager)
    return self._transform_input_tensor(input_tensor)

  @property
  def variable_shape(self):
    # [FUNCTION DOC]
    # Property returning the tensor shape equal to the column's configured shape (without the batch
    # dimension).
    """See `DenseColumn` base class."""
    return tensor_shape.TensorShape(self.shape)

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _variable_shape(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property; returns the same as variable_shape and emits a
    # deprecation warning.
    return self.variable_shape

  def get_dense_tensor(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns the dense tensor for the column. Inputs: transformation cache and state manager. Simply
    # looks up the already transformed value of this column in the cache (triggering the transform if
    # needed). Creates no variables.
    """Returns dense `Tensor` representing numeric feature.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Dense `Tensor` created within `transform_feature`.
    """
    # Feature has been already transformed. Return the intermediate
    # representation created by _transform_feature.
    return transformation_cache.get(self, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) dense tensor getter. Inputs: legacy input cache, plus weight
    # collections and trainable flag which are ignored. Returns the cached transformed tensor for this
    # column. Emits a deprecation warning.
    del weight_collections
    del trainable
    return inputs.get(self)

  @property
  def parents(self):
    # [FUNCTION DOC]
    # Property returning the list of inputs this column depends on: just the raw feature key.
    """See 'FeatureColumn` base class."""
    return [self.key]

  def get_config(self):
    # [FUNCTION DOC]
    # Serialises the column to a plain mapping from field names to values. The normalizer function is
    # serialized through the Keras object serializer and the dtype is replaced by its name. Returns the
    # mapping.
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['normalizer_fn'] = serialization._serialize_keras_object(  # pylint: disable=protected-access
        self.normalizer_fn)
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # Class method that rebuilds a NumericColumn from a configuration mapping. Inputs: the mapping,
    # optional custom objects used to deserialize the normalizer, and an unused columns-by-name mapping.
    # Validates that the mapping has exactly the expected field names (value error otherwise), copies it
    # turning lists into tuples (so shape is hashable), deserializes the normalizer, converts the dtype
    # name back to a dtype, and returns a new column instance.
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['normalizer_fn'] = serialization._deserialize_keras_object(  # pylint: disable=protected-access
        config['normalizer_fn'],
        custom_objects=custom_objects)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])

    return cls(**kwargs)


@serialization.register_feature_column
class BucketizedColumn(
    DenseColumn,
    CategoricalColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('BucketizedColumn',
                           ('source_column', 'boundaries'))):
  # [CLASS DOC]
  # Immutable named tuple (source_column, boundaries) produced by bucketized_column. Acts both as a
  # dense column (one-hot of the bucket index) and as a categorical column (bucket index as sparse
  # ids, offset per source dimension). The source column is a one-dimensional numeric column.
  # Registered for serialization.
  """See `bucketized_column`."""

  @property
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # Internal property: true when the source column is a v2 feature column that is itself v2.
    return (
        isinstance(self.source_column, fc_types.FeatureColumn)
        and self.source_column._is_v2_column
    )  # pylint: disable=protected-access

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: the source column's name with a bucketized suffix.
    """See `FeatureColumn` base class."""
    return '{}_bucketized'.format(self.source_column.name)

  @property
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # Property: the parsing specification is simply that of the source column.
    """See `FeatureColumn` base class."""
    return self.source_column.parse_example_spec

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property returning the source column's legacy parse specification.
    # Emits a deprecation warning.
    return self.source_column._parse_example_spec  # pylint: disable=protected-access

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) transform. Input: legacy input cache. Looks up the source column's
    # tensor and buckets it by the boundaries, returning integer bucket indices. Emits a deprecation
    # warning.
    """Returns bucketized categorical `source_column` tensor."""
    source_tensor = inputs.get(self.source_column)
    return math_ops._bucketize(  # pylint: disable=protected-access
        source_tensor,
        boundaries=self.boundaries)

  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Fetches the source column's tensor from the transformation cache and maps each value to its bucket
    # index using the stored boundaries (left inclusive). Inputs: cache and state manager. Returns an
    # integer tensor of bucket indices.
    """Returns bucketized categorical `source_column` tensor."""
    source_tensor = transformation_cache.get(self.source_column, state_manager)
    return math_ops._bucketize(  # pylint: disable=protected-access
        source_tensor,
        boundaries=self.boundaries)

  @property
  def variable_shape(self):
    # [FUNCTION DOC]
    # Property: shape of the one-hot output, namely the source column shape followed by the number of
    # boundaries plus one.
    """See `DenseColumn` base class."""
    return tensor_shape.TensorShape(
        tuple(self.source_column.shape) + (len(self.boundaries) + 1,))

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _variable_shape(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property equal to variable_shape; emits a deprecation warning.
    return self.variable_shape

  def _get_dense_tensor_for_input_tensor(self, input_tensor):
    # [FUNCTION DOC]
    # HELPER: Shared by the v1 and v2 dense paths. Input: a tensor of bucket indices. Returns a float
    # one-hot encoding (ones and zeros) with depth equal to the number of boundaries plus one.
    return array_ops.one_hot(
        indices=math_ops.cast(input_tensor, dtypes.int64),
        depth=len(self.boundaries) + 1,
        on_value=1.,
        off_value=0.)

  def get_dense_tensor(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns the one-hot encoded dense tensor. Inputs: cache and state manager. Gets the column's own
    # transformed bucket indices from the cache and one-hot encodes them.
    """Returns one hot encoded dense `Tensor`."""
    input_tensor = transformation_cache.get(self, state_manager)
    return self._get_dense_tensor_for_input_tensor(input_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) dense getter. Inputs: legacy input cache; weight collections and
    # trainable flag are ignored. Returns the one-hot encoding of the cached bucket indices. Emits a
    # deprecation warning.
    del weight_collections
    del trainable
    input_tensor = inputs.get(self)
    return self._get_dense_tensor_for_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    # [FUNCTION DOC]
    # Property: total number of ids when used as a categorical column, equal to the bucket count (boundaries
    # plus one) times the width of the source column, since each source dimension gets its own id range.
    """See `CategoricalColumn` base class."""
    # By construction, source_column is always one-dimensional.
    return (len(self.boundaries) + 1) * self.source_column.shape[0]

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property equal to num_buckets; emits a deprecation warning.
    return self.num_buckets

  def _get_sparse_tensors_for_input_tensor(self, input_tensor):
    # [FUNCTION DOC]
    # HELPER: Shared by the v1 and v2 sparse paths. Input: a batch by source-dimension tensor of bucket
    # indices. Builds a sparse tensor with one entry per element: indices are the (row, source
    # dimension) positions, values are the bucket index shifted by bucket count times the dimension
    # index so ids are unique across dimensions, and the dense shape is batch size by source dimension.
    # Returns an IdWeightPair of that sparse tensor and nothing for weights.
    batch_size = array_ops.shape(input_tensor)[0]
    # By construction, source_column is always one-dimensional.
    source_dimension = self.source_column.shape[0]

    i1 = array_ops.reshape(
        array_ops.tile(
            array_ops.expand_dims(math_ops.range(0, batch_size), 1),
            [1, source_dimension]), (-1,))
    i2 = array_ops.tile(math_ops.range(0, source_dimension), [batch_size])
    # Flatten the bucket indices and unique them across dimensions
    # E.g. 2nd dimension indices will range from k to 2*k-1 with k buckets
    bucket_indices = (
        array_ops.reshape(input_tensor,
                          (-1,)) + (len(self.boundaries) + 1) * i2)

    indices = math_ops.cast(
        array_ops.transpose(array_ops_stack.stack((i1, i2))), dtypes.int64)
    dense_shape = math_ops.cast(
        array_ops_stack.stack([batch_size, source_dimension]), dtypes.int64)
    sparse_tensor = sparse_tensor_lib.SparseTensor(
        indices=indices, values=bucket_indices, dense_shape=dense_shape)
    return CategoricalColumn.IdWeightPair(sparse_tensor, None)

  def get_sparse_tensors(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Converts the dense bucket indices of this column into the sparse id representation expected by
    # categorical consumers. Inputs: cache and state manager. Returns an IdWeightPair with no weights.
    """Converts dense inputs to SparseTensor so downstream code can use it."""
    input_tensor = transformation_cache.get(self, state_manager)
    return self._get_sparse_tensors_for_input_tensor(input_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self,
                          inputs,
                          weight_collections=None,
                          trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) sparse getter. Inputs: legacy input cache; weight collections and
    # trainable flag are ignored. Returns an IdWeightPair of the sparse conversion of the cached bucket
    # indices. Emits a deprecation warning.
    """Converts dense inputs to SparseTensor so downstream code can use it."""
    del weight_collections
    del trainable
    input_tensor = inputs.get(self)
    return self._get_sparse_tensors_for_input_tensor(input_tensor)

  @property
  def parents(self):
    # [FUNCTION DOC]
    # Property: list containing only the source column.
    """See 'FeatureColumn` base class."""
    return [self.source_column]

  def get_config(self):
    # [FUNCTION DOC]
    # Serialises the column into a mapping of field names to values, replacing the source column with
    # its own serialized configuration. The serialization module is imported lazily inside the method to
    # avoid a circular import. Returns the mapping.
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['source_column'] = serialize_feature_column(self.source_column)
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # Class method rebuilding a BucketizedColumn from a configuration mapping. Inputs: the mapping,
    # optional custom objects and an optional mapping from names to already revived columns (reused so
    # shared sub-columns stay identical). Validates the keys (value error on mismatch), copies the
    # configuration turning lists into tuples, deserializes the source column, and returns a new column.
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['source_column'] = deserialize_feature_column(
        config['source_column'], custom_objects, columns_by_name)
    return cls(**kwargs)


@serialization.register_feature_column
class EmbeddingColumn(
    DenseColumn,
    SequenceDenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    fc_old._SequenceDenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'EmbeddingColumn',
        ('categorical_column', 'dimension', 'combiner', 'initializer',
         'ckpt_to_load_from', 'tensor_name_in_ckpt', 'max_norm', 'trainable',
         'use_safe_embedding_lookup'))):
  # [CLASS DOC]
  # Immutable named tuple (categorical_column, dimension, combiner, initializer, ckpt_to_load_from,
  # tensor_name_in_ckpt, max_norm, trainable, use_safe_embedding_lookup) produced by the
  # embedding_column factory. Converts sparse categorical ids into a dense vector by looking them up
  # in an embedding variable created through a state manager. Works both as a dense and a sequence
  # dense column. Registered for serialization.
  """See `embedding_column`."""

  def __new__(cls,
              categorical_column,
              dimension,
              combiner,
              initializer,
              ckpt_to_load_from,
              tensor_name_in_ckpt,
              max_norm,
              trainable,
              use_safe_embedding_lookup=True):
    # [FUNCTION DOC]
    # Constructor of the named tuple. Inputs: categorical column, dimension, combiner, initializer,
    # checkpoint path, checkpoint tensor name, max norm, trainable flag and the safe-lookup flag
    # (default true). Simply forwards all values by name to the parent tuple constructor and returns
    # the new immutable instance; performs no validation.
    return super(EmbeddingColumn, cls).__new__(
        cls,
        categorical_column=categorical_column,
        dimension=dimension,
        combiner=combiner,
        initializer=initializer,
        ckpt_to_load_from=ckpt_to_load_from,
        tensor_name_in_ckpt=tensor_name_in_ckpt,
        max_norm=max_norm,
        trainable=trainable,
        use_safe_embedding_lookup=use_safe_embedding_lookup)

  @property
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # Internal property: true when the wrapped categorical column is a v2 feature column that is itself v2.
    return (
        isinstance(self.categorical_column, fc_types.FeatureColumn)
        and self.categorical_column._is_v2_column
    )  # pylint: disable=protected-access

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: the categorical column's name with an embedding suffix.
    """See `FeatureColumn` base class."""
    return '{}_embedding'.format(self.categorical_column.name)

  @property
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # Property: the parsing specification is that of the wrapped categorical column.
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property returning the wrapped column's legacy parse specification;
    # emits a deprecation warning.
    return self.categorical_column._parse_example_spec  # pylint: disable=protected-access

  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Transformation is delegated: returns the transformed output of the wrapped categorical column from
    # the cache. Inputs: cache and state manager.
    """Transforms underlying `categorical_column`."""
    return transformation_cache.get(self.categorical_column, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) transform; returns the wrapped column's tensor from the legacy input
    # cache; emits a deprecation warning.
    return inputs.get(self.categorical_column)

  @property
  def variable_shape(self):
    # [FUNCTION DOC]
    # Property: tensor shape with a single dimension equal to the embedding dimension.
    """See `DenseColumn` base class."""
    return tensor_shape.TensorShape([self.dimension])

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _variable_shape(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property equal to variable_shape; emits a deprecation warning.
    return self.variable_shape

  def create_state(self, state_manager):
    # [FUNCTION DOC]
    # Creates the embedding weight variable through the state manager (name embedding_weights, shape
    # bucket count by dimension, 32-bit float, resource variable, this column's trainable flag and
    # initializer). The bucket count is taken from the categorical column (v2 or legacy attribute).
    # Input: the state manager. Side effect: variable creation (raises a value error if it already
    # exists). Returns nothing.
    """Creates the embedding lookup variable."""
    default_num_buckets = (
        self.categorical_column.num_buckets
        if self._is_v2_column else self.categorical_column._num_buckets)  # pylint: disable=protected-access
    num_buckets = getattr(self.categorical_column, 'num_buckets',
                          default_num_buckets)
    embedding_shape = (num_buckets, self.dimension)
    state_manager.create_variable(
        self,
        name='embedding_weights',
        shape=embedding_shape,
        dtype=dtypes.float32,
        trainable=self.trainable,
        use_resource=True,
        initializer=self.initializer)

  def _get_dense_tensor_internal_helper(self, sparse_tensors,
                                        embedding_weights):
    # [FUNCTION DOC]
    # HELPER: Core embedding lookup shared by the v1 and v2 paths. Inputs: an IdWeightPair of sparse
    # tensors and the embedding weights (possibly partitioned). If a checkpoint was configured it
    # initialises the weights from it (using the variable list for partitioned variables) as a side
    # effect. Then performs a sparse embedding lookup with the column's combiner and max norm, using the
    # safe variant unless safe lookup is disabled and the id rank is statically known to be at most two,
    # in which case the faster plain variant is used. Returns the combined dense embedding tensor.
    sparse_ids = sparse_tensors.id_tensor
    sparse_weights = sparse_tensors.weight_tensor

    if self.ckpt_to_load_from is not None:
      to_restore = embedding_weights
      if isinstance(to_restore, variables.PartitionedVariable):
        to_restore = to_restore._get_variable_list()  # pylint: disable=protected-access
      checkpoint_utils.init_from_checkpoint(
          self.ckpt_to_load_from, {self.tensor_name_in_ckpt: to_restore})

    sparse_id_rank = tensor_shape.dimension_value(
        sparse_ids.dense_shape.get_shape()[0])
    embedding_lookup_sparse = embedding_ops.safe_embedding_lookup_sparse
    if (not self.use_safe_embedding_lookup and sparse_id_rank is not None and
        sparse_id_rank <= 2):
      embedding_lookup_sparse = embedding_ops.embedding_lookup_sparse_v2
    # Return embedding lookup result.
    return embedding_lookup_sparse(
        embedding_weights,
        sparse_ids,
        sparse_weights,
        combiner=self.combiner,
        name='%s_weights' % self.name,
        max_norm=self.max_norm)

  def _get_dense_tensor_internal(self, sparse_tensors, state_manager):
    # [FUNCTION DOC]
    # HELPER: Private v2 path. Inputs: sparse id/weight pair and state manager. Fetches the embedding
    # variable named embedding_weights from the state manager (error if missing) and returns the result
    # of the lookup helper.
    """Private method that follows the signature of get_dense_tensor."""
    embedding_weights = state_manager.get_variable(
        self, name='embedding_weights')
    return self._get_dense_tensor_internal_helper(sparse_tensors,
                                                  embedding_weights)

  def _old_get_dense_tensor_internal(self, sparse_tensors, weight_collections,
                                     trainable):
    # [FUNCTION DOC]
    # HELPER: Private legacy (v1) path. Inputs: sparse id/weight pair, a list of weight collections
    # (may be nothing) and a trainable flag. Appends the global-variables collection to the collection
    # list when absent (mutating the caller's list), gets or creates the embedding variable through the
    # variable scope (trainable only when both the column flag and the argument are true), and returns
    # the lookup helper result.
    """Private method that follows the signature of _get_dense_tensor."""
    embedding_shape = (self.categorical_column._num_buckets, self.dimension)  # pylint: disable=protected-access
    if (weight_collections and
        ops.GraphKeys.GLOBAL_VARIABLES not in weight_collections):
      weight_collections.append(ops.GraphKeys.GLOBAL_VARIABLES)
    embedding_weights = variable_scope.get_variable(
        name='embedding_weights',
        shape=embedding_shape,
        dtype=dtypes.float32,
        initializer=self.initializer,
        trainable=self.trainable and trainable,
        collections=weight_collections)
    return self._get_dense_tensor_internal_helper(sparse_tensors,
                                                  embedding_weights)

  def get_dense_tensor(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns the embedding lookup tensor for this column. Inputs: transformation cache and state
    # manager. Raises a value error when the wrapped column is a sequence categorical column (advice to
    # use sequence features is in the message). Otherwise gets sparse ids and weights from the wrapped
    # column and looks up embeddings with the variable from the state manager.
    """Returns tensor after doing the embedding lookup.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Embedding lookup tensor.

    Raises:
      ValueError: `categorical_column` is SequenceCategoricalColumn.
    """
    if isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must not be of type SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    # Get sparse IDs and weights.
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    return self._get_dense_tensor_internal(sparse_tensors, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) dense getter. Inputs: legacy input cache, optional weight
    # collections and trainable flag. Raises a value error for a (legacy or v2) sequence categorical
    # column. Otherwise fetches the sparse tensors from the wrapped column and does the legacy lookup,
    # creating the variable in the variable scope. Emits a deprecation warning.
    if isinstance(
        self.categorical_column,
        (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):  # pylint: disable=protected-access
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must not be of type _SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    sparse_tensors = self.categorical_column._get_sparse_tensors(  # pylint: disable=protected-access
        inputs, weight_collections, trainable)
    return self._old_get_dense_tensor_internal(sparse_tensors,
                                               weight_collections, trainable)

  def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns the embedded tensor and per-example sequence lengths for sequence input. Inputs: cache and
    # state manager. Raises a value error unless the wrapped column is a sequence categorical column.
    # Looks up embeddings for its sparse tensors and computes lengths from the sparse ids. Returns a
    # TensorSequenceLengthPair.
    """See `SequenceDenseColumn` base class."""
    if not isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must be of type SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    dense_tensor = self._get_dense_tensor_internal(sparse_tensors,
                                                   state_manager)
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sequence_dense_tensor(self,
                                 inputs,
                                 weight_collections=None,
                                 trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) version of the sequence getter. Inputs: legacy input cache, weight
    # collections and trainable flag. Raises a value error unless the wrapped column is a sequence
    # categorical column (v2 or legacy). Calls the legacy lookup and computes the sequence lengths from
    # the sparse ids. Returns a TensorSequenceLengthPair. Emits a deprecation warning.
    if not isinstance(
        self.categorical_column,
        (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):  # pylint: disable=protected-access
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must be of type SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)  # pylint: disable=protected-access
    dense_tensor = self._old_get_dense_tensor_internal(
        sparse_tensors,
        weight_collections=weight_collections,
        trainable=trainable)
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  @property
  def parents(self):
    # [FUNCTION DOC]
    # Property: list containing only the wrapped categorical column.
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]

  def get_config(self):
    # [FUNCTION DOC]
    # Serialises the column into a mapping from field names to values, replacing the categorical column
    # by its serialized form and the initializer by its Keras serialization. Returns the mapping.
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['categorical_column'] = serialization.serialize_feature_column(
        self.categorical_column)
    config['initializer'] = serialization._serialize_keras_object(  # pylint: disable=protected-access
        self.initializer)
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # Class method rebuilding an EmbeddingColumn from a configuration mapping. Inputs: the mapping,
    # optional custom objects, optional mapping of already revived columns. Adds the safe-lookup flag
    # (true) to older configs lacking it (mutating the passed mapping), validates keys (value error on
    # mismatch), converts lists to tuples, deserializes the categorical column and the initializer (looked
    # up among the initializer classes of the init ops module and custom objects). Returns a new column.
    """See 'FeatureColumn` base class."""
    if 'use_safe_embedding_lookup' not in config:
      config['use_safe_embedding_lookup'] = True
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['categorical_column'] = serialization.deserialize_feature_column(
        config['categorical_column'], custom_objects, columns_by_name)
    all_initializers = dict(tf_inspect.getmembers(init_ops, tf_inspect.isclass))
    kwargs['initializer'] = serialization._deserialize_keras_object(  # pylint: disable=protected-access
        config['initializer'],
        module_objects=all_initializers,
        custom_objects=custom_objects)
    return cls(**kwargs)


def _raise_shared_embedding_column_error():
  # [FUNCTION DOC]
  # HELPER: Guard used by the legacy entry points of the shared embedding column. Always raises a
  # value error explaining that shared embedding columns are not supported by the legacy linear model
  # or input layer and that the new dense features layer or linear model should be used.
  raise ValueError('SharedEmbeddingColumns are not supported in '
                   '`linear_model` or `input_layer`. Please use '
                   '`DenseFeatures` or `LinearModel` instead.')


class SharedEmbeddingColumnCreator(autotrackable.AutoTrackable):
  # [CLASS DOC]
  # Trackable factory object that owns the embedding weights shared by a group of columns and
  # creates the SharedEmbeddingColumn instances that use them. Holds the dimension, initializer,
  # checkpoint path and tensor name, bucket count, trainable flag, variable name, safe-lookup flag and a
  # mapping from graph key to the created weight variable (so each graph gets its own variable).
  """Class that creates a `SharedEmbeddingColumn`."""

  def __init__(self,
               dimension,
               initializer,
               ckpt_to_load_from,
               tensor_name_in_ckpt,
               num_buckets,
               trainable,
               name='shared_embedding_column_creator',
               use_safe_embedding_lookup=True):
    # [FUNCTION DOC]
    # Stores the configuration. Inputs: embedding dimension, initializer, checkpoint path (or nothing),
    # checkpoint tensor name (or nothing), number of buckets, trainable flag, variable name (default
    # a shared column creator name) and the safe-lookup flag (default true). Creates an empty mapping
    # from graph key to variable. No variables are created yet. Returns nothing.
    self._dimension = dimension
    self._initializer = initializer
    self._ckpt_to_load_from = ckpt_to_load_from
    self._tensor_name_in_ckpt = tensor_name_in_ckpt
    self._num_buckets = num_buckets
    self._trainable = trainable
    self._name = name
    self._use_safe_embedding_lookup = use_safe_embedding_lookup
    # Map from graph keys to embedding_weight variables.
    self._embedding_weights = {}

  def __call__(self, categorical_column, combiner, max_norm):
    # [FUNCTION DOC]
    # Makes the creator callable. Inputs: a categorical column, a combiner name and a max norm. Returns a
    # new SharedEmbeddingColumn bound to this creator and its safe-lookup setting.
    return SharedEmbeddingColumn(categorical_column, self, combiner, max_norm,
                                 self._use_safe_embedding_lookup)

  @property
  def embedding_weights(self):
    # [FUNCTION DOC]
    # Property returning the shared weight variable for the current default graph. On first access in a
    # graph, creates the variable (shape bucket count by dimension, float32, the stored initializer and
    # trainable flag, under the stored name) with the variable scope and, if a checkpoint was given,
    # registers initialisation from the checkpoint (using the variable list for partitioned variables).
    # The result is cached per graph key and returned on later accesses.
    key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    if key not in self._embedding_weights:
      embedding_shape = (self._num_buckets, self._dimension)
      var = variable_scope.get_variable(
          name=self._name,
          shape=embedding_shape,
          dtype=dtypes.float32,
          initializer=self._initializer,
          trainable=self._trainable)

      if self._ckpt_to_load_from is not None:
        to_restore = var
        if isinstance(to_restore, variables.PartitionedVariable):
          to_restore = to_restore._get_variable_list()  # pylint: disable=protected-access
        checkpoint_utils.init_from_checkpoint(
            self._ckpt_to_load_from, {self._tensor_name_in_ckpt: to_restore})
      self._embedding_weights[key] = var
    return self._embedding_weights[key]

  @property
  def dimension(self):
    # [FUNCTION DOC]
    # Property returning the embedding dimension of the shared weights.
    return self._dimension


@serialization.register_feature_column
class SharedEmbeddingColumn(
    DenseColumn,
    SequenceDenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    fc_old._SequenceDenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'SharedEmbeddingColumn',
        ('categorical_column', 'shared_embedding_column_creator', 'combiner',
         'max_norm', 'use_safe_embedding_lookup'))):
  # [CLASS DOC]
  # Immutable named tuple (categorical_column, shared_embedding_column_creator, combiner, max_norm,
  # use_safe_embedding_lookup) for an embedding column whose weights live in a shared creator object.
  # Supported only through the v2 code paths; the legacy underscore entry points all raise errors.
  # Works both as a dense and a sequence dense column. Registered for serialization.
  """See `embedding_column`."""

  def __new__(cls,
              categorical_column,
              shared_embedding_column_creator,
              combiner,
              max_norm,
              use_safe_embedding_lookup=True):
    # [FUNCTION DOC]
    # Constructor of the named tuple. Inputs: categorical column, creator, combiner, max norm and the
    # safe-lookup flag (default true). Forwards all values by name to the parent constructor; no
    # validation. Returns the new immutable instance.
    return super(SharedEmbeddingColumn, cls).__new__(
        cls,
        categorical_column=categorical_column,
        shared_embedding_column_creator=shared_embedding_column_creator,
        combiner=combiner,
        max_norm=max_norm,
        use_safe_embedding_lookup=use_safe_embedding_lookup)

  @property
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # Internal property marking this as a v2 column; always true.
    return True

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: the categorical column's name with a shared embedding suffix.
    """See `FeatureColumn` base class."""
    return '{}_shared_embedding'.format(self.categorical_column.name)

  @property
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # Property: parsing specification of the wrapped categorical column.
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  @property
  def _parse_example_spec(self):
    # [FUNCTION DOC]
    # HELPER: Legacy (v1) property that is unsupported; always raises the shared embedding value error.
    return _raise_shared_embedding_column_error()

  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Delegates the transformation to the wrapped categorical column via the cache. Inputs: cache and
    # state manager. Returns its transformed tensor.
    """See `FeatureColumn` base class."""
    return transformation_cache.get(self.categorical_column, state_manager)

  def _transform_feature(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Legacy (v1) transform that is unsupported; always raises the shared embedding value error.
    return _raise_shared_embedding_column_error()

  @property
  def variable_shape(self):
    # [FUNCTION DOC]
    # Property: tensor shape with one dimension equal to the shared embedding dimension.
    """See `DenseColumn` base class."""
    return tensor_shape.TensorShape(
        [self.shared_embedding_column_creator.dimension])

  @property
  def _variable_shape(self):
    # [FUNCTION DOC]
    # HELPER: Legacy (v1) property that is unsupported; always raises the shared embedding value error.
    return _raise_shared_embedding_column_error()

  def _get_dense_tensor_internal(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # HELPER: Core lookup for dense and sequence use. Inputs: cache and state manager. Opens a name
    # scope named after the column so ops of different shared columns have distinct names, gets sparse
    # ids and weights from the wrapped column, obtains the shared weights from the creator for the
    # current graph, and performs the sparse embedding lookup with this column's combiner and max norm
    # (safe variant unless disabled and the id rank is statically at most two). Returns the embedding
    # tensor. Side effect: may create the shared variable on first use.
    """Private method that follows the signature of _get_dense_tensor."""
    # This method is called from a variable_scope with name _var_scope_name,
    # which is shared among all shared embeddings. Open a name_scope here, so
    # that the ops for different columns have distinct names.
    with ops.name_scope(None, default_name=self.name):
      # Get sparse IDs and weights.
      sparse_tensors = self.categorical_column.get_sparse_tensors(
          transformation_cache, state_manager)
      sparse_ids = sparse_tensors.id_tensor
      sparse_weights = sparse_tensors.weight_tensor

      embedding_weights = self.shared_embedding_column_creator.embedding_weights

      sparse_id_rank = tensor_shape.dimension_value(
          sparse_ids.dense_shape.get_shape()[0])
      embedding_lookup_sparse = embedding_ops.safe_embedding_lookup_sparse
      if (not self.use_safe_embedding_lookup and sparse_id_rank is not None and
          sparse_id_rank <= 2):
        embedding_lookup_sparse = embedding_ops.embedding_lookup_sparse_v2
      # Return embedding lookup result.
      return embedding_lookup_sparse(
          embedding_weights,
          sparse_ids,
          sparse_weights,
          combiner=self.combiner,
          name='%s_weights' % self.name,
          max_norm=self.max_norm)

  def get_dense_tensor(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns the embedding lookup result. Inputs: cache and state manager. Raises a value error if the
    # wrapped column is a sequence categorical column. Otherwise delegates to the internal lookup.
    """Returns the embedding lookup result."""
    if isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must not be of type SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    return self._get_dense_tensor_internal(transformation_cache, state_manager)

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    # [FUNCTION DOC]
    # HELPER: Legacy (v1) dense getter that is unsupported; always raises the shared embedding value error.
    return _raise_shared_embedding_column_error()

  def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns embeddings and sequence lengths for sequence input. Inputs: cache and state manager.
    # Raises a value error unless the wrapped column is a sequence categorical column. Computes the
    # embedding tensor, then the lengths from the wrapped column's sparse ids, and returns a
    # TensorSequenceLengthPair.
    """See `SequenceDenseColumn` base class."""
    if not isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must be of type SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    dense_tensor = self._get_dense_tensor_internal(transformation_cache,
                                                   state_manager)
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  def _get_sequence_dense_tensor(self,
                                 inputs,
                                 weight_collections=None,
                                 trainable=None):
    # [FUNCTION DOC]
    # HELPER: Legacy (v1) sequence getter that is unsupported; always raises the shared embedding value error.
    return _raise_shared_embedding_column_error()

  @property
  def parents(self):
    # [FUNCTION DOC]
    # Property: list containing only the wrapped categorical column.
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]


def _check_shape(shape, key):
  # [FUNCTION DOC]
  # HELPER: Validation utility for numeric_column. Input: a shape (single integer or iterable of
  # integers; must not be nothing, asserted) and the feature key for messages. Wraps a scalar into a
  # list, converts to a tuple, and checks each dimension. Raises a type error for a non-integer
  # dimension and a value error for a dimension below 1. Returns the shape tuple.
  """Returns shape if it's valid, raises error otherwise."""
  assert shape is not None
  if not nest.is_nested(shape):
    shape = [shape]
  shape = tuple(shape)
  for dimension in shape:
    if not isinstance(dimension, int):
      raise TypeError('shape dimensions must be integer. '
                      'shape: {}, key: {}'.format(shape, key))
    if dimension < 1:
      raise ValueError('shape dimensions must be greater than 0. '
                       'shape: {}, key: {}'.format(shape, key))
  return shape


@serialization.register_feature_column
class HashedCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('HashedCategoricalColumn',
                           ('key', 'hash_bucket_size', 'dtype'))):
  # [CLASS DOC]
  # Immutable named tuple (key, hash_bucket_size, dtype) produced by categorical_column_with_hash_bucket.
  # Reads a sparse string or integer feature and hashes each value into a fixed number of buckets.
  # Registered for serialization.
  """see `categorical_column_with_hash_bucket`."""

  @property
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # Internal property marking this as a v2 column; always true.
    return True

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: the column name, equal to the feature key.
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # Property: one-entry mapping from the key to a variable-length feature of the column dtype.
    """See `FeatureColumn` base class."""
    return {self.key: parsing_ops.VarLenFeature(self.dtype)}

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property equal to parse_example_spec; emits a deprecation warning.
    return self.parse_example_spec

  def _transform_input_tensor(self, input_tensor):
    # [FUNCTION DOC]
    # HELPER: Shared hashing logic. Input: a sparse tensor. Raises a value error if the input is not
    # sparse, if its dtype is neither string nor integer (via assertion helper), or if its integer-ness
    # does not match the column dtype. Integer values are converted to strings first; strings are then
    # hashed into the configured number of buckets with the fast string hash. Returns a sparse tensor
    # with the same indices and dense shape and the bucket ids as values.
    """Hashes the values in the feature_column."""
    if not isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
      raise ValueError('SparseColumn input must be a SparseTensor.')

    fc_utils.assert_string_or_int(
        input_tensor.dtype,
        prefix='column_name: {} input_tensor'.format(self.key))

    if self.dtype.is_integer != input_tensor.dtype.is_integer:
      raise ValueError(
          'Column dtype and SparseTensors dtype must be compatible. '
          'key: {}, column dtype: {}, tensor dtype: {}'.format(
              self.key, self.dtype, input_tensor.dtype))

    if self.dtype == dtypes.string:
      sparse_values = input_tensor.values
    else:
      sparse_values = string_ops.as_string(input_tensor.values)

    sparse_id_values = string_ops.string_to_hash_bucket_fast(
        sparse_values, self.hash_bucket_size, name='lookup')
    return sparse_tensor_lib.SparseTensor(input_tensor.indices,
                                          sparse_id_values,
                                          input_tensor.dense_shape)

  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Hashes the column's raw feature. Inputs: cache and state manager. Fetches the raw tensor by key,
    # converts it to a sparse tensor dropping ignore values, and hashes it. Returns the sparse id tensor.
    """Hashes the values in the feature_column."""
    input_tensor = _to_sparse_input_and_drop_ignore_values(
        transformation_cache.get(self.key, state_manager))
    return self._transform_input_tensor(input_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) transform. Input: legacy input cache. Same as transform_feature but
    # reads from the legacy cache. Emits a deprecation warning.
    input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
    return self._transform_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    # [FUNCTION DOC]
    # Property: the number of ids, equal to the hash bucket size.
    """Returns number of buckets in this sparse feature."""
    return self.hash_bucket_size

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property equal to num_buckets; emits a deprecation warning.
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns an IdWeightPair whose ids are the column's transformed (hashed) sparse tensor from the
    # cache and whose weights are nothing. Inputs: cache and state manager.
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self,
                          inputs,
                          weight_collections=None,
                          trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) getter. Inputs: legacy input cache; weight collections and trainable
    # flag are ignored. Returns an IdWeightPair of the cached transformed tensor and nothing. Emits a
    # deprecation warning.
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    # [FUNCTION DOC]
    # Property: list containing only the raw feature key.
    """See 'FeatureColumn` base class."""
    return [self.key]

  def get_config(self):
    # [FUNCTION DOC]
    # Serialises to a mapping of field names to values with the dtype replaced by its name. Returns the mapping.
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # Class method rebuilding the column from a configuration mapping. Inputs: the mapping, plus
    # optional custom objects and columns-by-name which are unused. Validates the keys (value error on
    # mismatch), converts lists to tuples, converts the dtype name back to a dtype, returns a new column.
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)


@serialization.register_feature_column
class VocabularyFileCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'VocabularyFileCategoricalColumn',
        ('key', 'vocabulary_file', 'vocabulary_size', 'num_oov_buckets',
         'dtype', 'default_value', 'file_format'))):
  # [CLASS DOC]
  # Immutable named tuple (key, vocabulary_file, vocabulary_size, num_oov_buckets, dtype,
  # default_value, file_format) produced by categorical_column_with_vocabulary_file. Maps string or
  # integer values to line numbers of a vocabulary file using a lookup table, with optional hashed
  # out-of-vocabulary buckets. Tables are cached as resources in the state manager. Registered for
  # serialization.
  """See `categorical_column_with_vocabulary_file`."""

  def __new__(cls,
              key,
              vocabulary_file,
              vocabulary_size,
              num_oov_buckets,
              dtype,
              default_value,
              file_format=None):
    # [FUNCTION DOC]
    # Constructor of the named tuple. Inputs: key, vocabulary file, vocabulary size, number of
    # out-of-vocabulary buckets, dtype, default value and an optional file format (default nothing).
    # Forwards values by name to the parent constructor; no validation. Returns the new instance.
    return super(VocabularyFileCategoricalColumn, cls).__new__(
        cls,
        key=key,
        vocabulary_file=vocabulary_file,
        vocabulary_size=vocabulary_size,
        num_oov_buckets=num_oov_buckets,
        dtype=dtype,
        default_value=default_value,
        file_format=file_format)

  @property
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # Internal property marking this as a v2 column; always true.
    return True

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: the column name, equal to the feature key.
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # Property: one-entry mapping from the key to a variable-length feature of the column dtype.
    """See `FeatureColumn` base class."""
    return {self.key: parsing_ops.VarLenFeature(self.dtype)}

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property equal to parse_example_spec; emits a deprecation warning.
    return self.parse_example_spec

  def _make_table_from_tfrecord_gzip_file(self, key_dtype, name):
    # [FUNCTION DOC]
    # HELPER: Builds the lookup table when the vocabulary is stored in a gzip-compressed record file.
    # Inputs: the key dtype and a table name. Opens the file as a dataset with gzip compression, maps
    # each record to the key dtype (strings kept, others parsed as numbers using a nested converter),
    # and creates an index table from the dataset with the configured oov bucket count, vocabulary
    # size, default value and name. Returns the table.
    dataset = readers.TFRecordDataset(
        self.vocabulary_file, compression_type='GZIP')

    def key_dtype_fn(key):
      # [FUNCTION DOC]
      # HELPER: Nested converter applied to each dataset record. Input: one record string. Returns it
      # unchanged if the key dtype is string, otherwise converts the string to a number of the key dtype.
      return key if key_dtype is dtypes.string else string_ops.string_to_number(
          key, out_type=key_dtype)

    return data_lookup_ops.index_table_from_dataset(
        dataset.map(key_dtype_fn),
        num_oov_buckets=self.num_oov_buckets,
        vocab_size=self.vocabulary_size,
        default_value=self.default_value,
        key_dtype=key_dtype,
        name=name)

  def _make_table(self, key_dtype, state_manager):
    # [FUNCTION DOC]
    # HELPER: Gets or creates the vocabulary lookup table. Inputs: the key dtype and a state manager
    # (may be nothing). The table name is the key plus a lookup suffix. If there is no state manager or it
    # has no such resource, creates the table inside an init scope (so it lives outside any function
    # graph), choosing between the gzip record file path and a text-file index table, and registers it
    # with the state manager when one exists. Otherwise reuses the stored table. Returns the table.
    name = '{}_lookup'.format(self.key)
    if state_manager is None or not state_manager.has_resource(self, name):
      with ops.init_scope():
        if self.file_format == 'tfrecord_gzip':
          table = self._make_table_from_tfrecord_gzip_file(key_dtype, name)
        else:
          table = lookup_ops.index_table_from_file(
              vocabulary_file=self.vocabulary_file,
              num_oov_buckets=self.num_oov_buckets,
              vocab_size=self.vocabulary_size,
              default_value=self.default_value,
              key_dtype=key_dtype,
              name=name)
      if state_manager is not None:
        state_manager.add_resource(self, name, table)
    else:
      # Reuse the table from the previous run.
      table = state_manager.get_resource(self, name)
    return table

  def _transform_input_tensor(self, input_tensor, state_manager=None):
    # [FUNCTION DOC]
    # HELPER: Shared lookup logic. Inputs: a sparse tensor and an optional state manager. Raises value
    # errors when the tensor dtype is not string or integer or mismatches the column dtype's
    # integer-ness. Integer inputs are cast to 64-bit integers (and the table key type becomes 64-bit).
    # Looks the values up in the table. Returns a sparse tensor of ids.
    """Creates a lookup table for the vocabulary."""
    if self.dtype.is_integer != input_tensor.dtype.is_integer:
      raise ValueError(
          'Column dtype and SparseTensors dtype must be compatible. '
          'key: {}, column dtype: {}, tensor dtype: {}'.format(
              self.key, self.dtype, input_tensor.dtype))

    fc_utils.assert_string_or_int(
        input_tensor.dtype,
        prefix='column_name: {} input_tensor'.format(self.key))

    key_dtype = self.dtype
    if input_tensor.dtype.is_integer:
      # `index_table_from_file` requires 64-bit integer keys.
      key_dtype = dtypes.int64
      input_tensor = math_ops.cast(input_tensor, dtypes.int64)
    return self._make_table(key_dtype, state_manager).lookup(input_tensor)

  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Looks up the raw feature in the vocabulary. Inputs: cache and state manager. Fetches the raw
    # tensor, converts to sparse dropping ignore values, and performs the table lookup (table kept in
    # the state manager). Returns a sparse id tensor.
    """Creates a lookup table for the vocabulary."""
    input_tensor = _to_sparse_input_and_drop_ignore_values(
        transformation_cache.get(self.key, state_manager))
    return self._transform_input_tensor(input_tensor, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) transform. Input: legacy input cache. Same as transform_feature but
    # passes no state manager, so a fresh table is created on every call. Emits a deprecation warning.
    input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
    return self._transform_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    # [FUNCTION DOC]
    # Property: total ids, which is the vocabulary size plus the number of oov buckets.
    """Returns number of buckets in this sparse feature."""
    return self.vocabulary_size + self.num_oov_buckets

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property equal to num_buckets; emits a deprecation warning.
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns an IdWeightPair with the column's transformed id sparse tensor from the cache and no
    # weights. Inputs: cache and state manager.
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self,
                          inputs,
                          weight_collections=None,
                          trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) getter. Inputs: legacy input cache; weight collections and trainable
    # flag are ignored. Returns an IdWeightPair of the cached transformed tensor and nothing. Emits a
    # deprecation warning.
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    # [FUNCTION DOC]
    # Property: list containing only the raw feature key.
    """See 'FeatureColumn` base class."""
    return [self.key]

  def get_config(self):
    # [FUNCTION DOC]
    # Serialises to a mapping of field names to values with the dtype replaced by its name. Returns the mapping.
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # Class method rebuilding the column from a configuration mapping. Inputs: mapping plus unused
    # custom objects and columns-by-name. Validates keys (value error on mismatch), converts lists to
    # tuples, converts the dtype name to a dtype, returns a new column.
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)


@serialization.register_feature_column
class VocabularyListCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'VocabularyListCategoricalColumn',
        ('key', 'vocabulary_list', 'dtype', 'default_value', 'num_oov_buckets'))
):
  # [CLASS DOC]
  # Immutable named tuple (key, vocabulary_list, dtype, default_value, num_oov_buckets) produced by
  # categorical_column_with_vocabulary_list. Maps values to their index in an in-memory vocabulary
  # via a lookup table built from a tensor, with optional hashed out-of-vocabulary buckets. Tables are
  # cached as resources in the state manager. Registered for serialization.
  """See `categorical_column_with_vocabulary_list`."""

  @property
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # Internal property marking this as a v2 column; always true.
    return True

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: the column name, equal to the feature key.
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # Property: one-entry mapping from the key to a variable-length feature of the column dtype.
    """See `FeatureColumn` base class."""
    return {self.key: parsing_ops.VarLenFeature(self.dtype)}

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property equal to parse_example_spec; emits a deprecation warning.
    return self.parse_example_spec

  def _transform_input_tensor(self, input_tensor, state_manager=None):
    # [FUNCTION DOC]
    # HELPER: Shared lookup logic. Inputs: a sparse tensor and an optional state manager. Raises value
    # errors for a dtype that is not string or integer or whose integer-ness mismatches the column
    # dtype. Integer inputs are cast to 64-bit. Gets the lookup table from the state manager or builds a
    # new one from the vocabulary tuple (inside an init scope) with the default value and oov bucket
    # count, registering it with the state manager when present. Returns the looked-up sparse ids.
    """Creates a lookup table for the vocabulary list."""
    if self.dtype.is_integer != input_tensor.dtype.is_integer:
      raise ValueError(
          'Column dtype and SparseTensors dtype must be compatible. '
          'key: {}, column dtype: {}, tensor dtype: {}'.format(
              self.key, self.dtype, input_tensor.dtype))

    fc_utils.assert_string_or_int(
        input_tensor.dtype,
        prefix='column_name: {} input_tensor'.format(self.key))

    key_dtype = self.dtype
    if input_tensor.dtype.is_integer:
      # `index_table_from_tensor` requires 64-bit integer keys.
      key_dtype = dtypes.int64
      input_tensor = math_ops.cast(input_tensor, dtypes.int64)

    name = '{}_lookup'.format(self.key)
    if state_manager is None or not state_manager.has_resource(self, name):
      with ops.init_scope():
        table = lookup_ops.index_table_from_tensor(
            vocabulary_list=tuple(self.vocabulary_list),
            default_value=self.default_value,
            num_oov_buckets=self.num_oov_buckets,
            dtype=key_dtype,
            name=name)
      if state_manager is not None:
        state_manager.add_resource(self, name, table)
    else:
      # Reuse the table from the previous run.
      table = state_manager.get_resource(self, name)
    return table.lookup(input_tensor)

  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Looks up the raw feature in the vocabulary list. Inputs: cache and state manager. Fetches the raw
    # tensor, makes it sparse dropping ignore values and performs the lookup using the state manager
    # for table caching. Returns a sparse id tensor.
    """Creates a lookup table for the vocabulary list."""
    input_tensor = _to_sparse_input_and_drop_ignore_values(
        transformation_cache.get(self.key, state_manager))
    return self._transform_input_tensor(input_tensor, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) transform. Input: legacy input cache. Same as transform_feature but
    # passes no state manager (a new table each call). Emits a deprecation warning.
    input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
    return self._transform_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    # [FUNCTION DOC]
    # Property: total ids, equal to the vocabulary length plus the number of oov buckets.
    """Returns number of buckets in this sparse feature."""
    return len(self.vocabulary_list) + self.num_oov_buckets

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property equal to num_buckets; emits a deprecation warning.
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns an IdWeightPair with the column's transformed id sparse tensor from the cache and no
    # weights. Inputs: cache and state manager.
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self,
                          inputs,
                          weight_collections=None,
                          trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) getter. Inputs: legacy input cache; weight collections and trainable
    # flag are ignored. Returns an IdWeightPair of the cached transformed tensor and nothing. Emits a
    # deprecation warning.
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    # [FUNCTION DOC]
    # Property: list containing only the raw feature key.
    """See 'FeatureColumn` base class."""
    return [self.key]

  def get_config(self):
    # [FUNCTION DOC]
    # Serialises to a mapping of field names to values with the dtype replaced by its name. Returns the mapping.
    """See 'FeatureColumn` base class."""
    config = dict(zip(self._fields, self))
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # Class method rebuilding the column from a configuration mapping. Inputs: mapping plus unused
    # custom objects and columns-by-name. Validates keys (value error on mismatch), converts lists to
    # tuples (the vocabulary becomes a tuple), converts the dtype name to a dtype, returns a new column.
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)


@serialization.register_feature_column
class IdentityCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('IdentityCategoricalColumn',
                           ('key', 'number_buckets', 'default_value'))):
  # [CLASS DOC]
  # Immutable named tuple (key, number_buckets, default_value) produced by
  # categorical_column_with_identity. Uses an integer input value directly as the category id, with an
  # optional default id for out-of-range values. Registered for serialization.
  """See `categorical_column_with_identity`."""

  @property
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # Internal property marking this as a v2 column; always true.
    return True

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: the column name, equal to the feature key.
    """See `FeatureColumn` base class."""
    return self.key

  @property
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # Property: one-entry mapping from the key to a variable-length 64-bit integer feature.
    """See `FeatureColumn` base class."""
    return {self.key: parsing_ops.VarLenFeature(dtypes.int64)}

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property equal to parse_example_spec; emits a deprecation warning.
    return self.parse_example_spec

  def _transform_input_tensor(self, input_tensor):
    # [FUNCTION DOC]
    # HELPER: Shared id logic for the v1 and v2 transforms.
    # KNOWN ISSUE: When no default value is set, nothing here checks that values are below the bucket
    # count, although the public docs say such values cause a failure; out-of-range ids pass through.
    # Input: a sparse tensor. Raises a value error when its dtype is not
    # integer. Casts values to 64-bit integers; when a default value is configured, every value below
    # zero or at or above the bucket count is replaced by the default (the values are cast a second
    # time from the original tensor, which is redundant). Returns a sparse tensor with the same indices
    # and dense shape.
    """Returns a SparseTensor with identity values."""
    if not input_tensor.dtype.is_integer:
      raise ValueError('Invalid input, not integer. key: {} dtype: {}'.format(
          self.key, input_tensor.dtype))
    values = input_tensor.values
    if input_tensor.values.dtype != dtypes.int64:
      values = math_ops.cast(values, dtypes.int64, name='values')
    if self.default_value is not None:
      values = math_ops.cast(input_tensor.values, dtypes.int64, name='values')
      num_buckets = math_ops.cast(
          self.num_buckets, dtypes.int64, name='num_buckets')
      zero = math_ops.cast(0, dtypes.int64, name='zero')
      # Assign default for out-of-range values.
      values = array_ops.where_v2(
          math_ops.logical_or(
              values < zero, values >= num_buckets, name='out_of_range'),
          array_ops.fill(
              dims=array_ops.shape(values),
              value=math_ops.cast(self.default_value, dtypes.int64),
              name='default_values'), values)

    return sparse_tensor_lib.SparseTensor(
        indices=input_tensor.indices,
        values=values,
        dense_shape=input_tensor.dense_shape)

  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Produces the id tensor. Inputs: cache and state manager. Fetches the raw feature, converts to a
    # sparse tensor dropping ignore values, then applies the identity/default logic. Returns a sparse
    # id tensor.
    """Returns a SparseTensor with identity values."""
    input_tensor = _to_sparse_input_and_drop_ignore_values(
        transformation_cache.get(self.key, state_manager))
    return self._transform_input_tensor(input_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) transform. Input: legacy input cache. Same as transform_feature
    # reading from the legacy cache. Emits a deprecation warning.
    input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
    return self._transform_input_tensor(input_tensor)

  @property
  def num_buckets(self):
    # [FUNCTION DOC]
    # Property: number of ids, equal to the configured number of buckets.
    """Returns number of buckets in this sparse feature."""
    return self.number_buckets

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property equal to num_buckets; emits a deprecation warning.
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns an IdWeightPair with the column's transformed id sparse tensor from the cache and no
    # weights. Inputs: cache and state manager.
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self,
                          inputs,
                          weight_collections=None,
                          trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) getter. Inputs: legacy input cache; weight collections and trainable
    # flag are ignored. Returns an IdWeightPair of the cached transformed tensor and nothing. Emits a
    # deprecation warning.
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    # [FUNCTION DOC]
    # Property: list containing only the raw feature key.
    """See 'FeatureColumn` base class."""
    return [self.key]

  def get_config(self):
    # [FUNCTION DOC]
    # Serialises to a mapping of field names to values; no conversion is needed since all fields are
    # plain values. Returns the mapping.
    """See 'FeatureColumn` base class."""
    return dict(zip(self._fields, self))

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # Class method rebuilding the column from a configuration mapping. Inputs: mapping plus unused
    # custom objects and columns-by-name. Validates keys (value error on mismatch), converts lists to
    # tuples and returns a new column.
    """See 'FeatureColumn` base class."""
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    return cls(**kwargs)


@serialization.register_feature_column
class WeightedCategoricalColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'WeightedCategoricalColumn',
        ('categorical_column', 'weight_feature_key', 'dtype'))):
  # [CLASS DOC]
  # Immutable named tuple (categorical_column, weight_feature_key, dtype) produced by
  # weighted_categorical_column. Combines the ids of a wrapped categorical column with a parallel
  # sparse weight feature, yielding id/weight pairs. Registered for serialization.
  """See `weighted_categorical_column`."""

  @property
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # Internal property: true when the wrapped categorical column is a v2 feature column that is itself v2.
    return (
        isinstance(self.categorical_column, fc_types.FeatureColumn)
        and self.categorical_column._is_v2_column
    )  # pylint: disable=protected-access

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: the wrapped column's name followed by a suffix naming the weight feature key.
    """See `FeatureColumn` base class."""
    return '{}_weighted_by_{}'.format(self.categorical_column.name,
                                      self.weight_feature_key)

  @property
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Adds the weight entry directly into the mapping returned by the wrapped column; if
    # that column returns a shared mapping rather than a fresh copy, the shared mapping is mutated.
    # Property: the wrapped column's parse specification extended with a variable-length feature for the
    # weight key (column dtype). Raises a value error when the weight key already has an entry.
    """See `FeatureColumn` base class."""
    config = self.categorical_column.parse_example_spec
    if self.weight_feature_key in config:
      raise ValueError('Parse config {} already exists for {}.'.format(
          config[self.weight_feature_key], self.weight_feature_key))
    config[self.weight_feature_key] = parsing_ops.VarLenFeature(self.dtype)
    return config

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property. Same as parse_example_spec but based on the wrapped
    # column's legacy specification; raises a value error on a duplicate weight key. Emits a deprecation
    # warning.
    config = self.categorical_column._parse_example_spec  # pylint: disable=protected-access
    if self.weight_feature_key in config:
      raise ValueError('Parse config {} already exists for {}.'.format(
          config[self.weight_feature_key], self.weight_feature_key))
    config[self.weight_feature_key] = parsing_ops.VarLenFeature(self.dtype)
    return config

  @property
  def num_buckets(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring refers to the DenseColumn base class, but this is a categorical
    # property.
    # Property: number of buckets of the wrapped categorical column.
    """See `DenseColumn` base class."""
    return self.categorical_column.num_buckets

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property returning the wrapped column's legacy bucket count; emits a
    # deprecation warning.
    return self.categorical_column._num_buckets  # pylint: disable=protected-access

  def _transform_weight_tensor(self, weight_tensor):
    # [FUNCTION DOC]
    # HELPER: Validates and converts the weight feature. Input: the raw weight tensor (may be nothing).
    # Raises a value error when it is missing or its base dtype differs from the column dtype. Converts
    # to a tensor or sparse tensor; dense tensors are made sparse dropping zero entries; non-floating
    # weights are cast to 32-bit float. Returns a sparse weight tensor.
    if weight_tensor is None:
      raise ValueError('Missing weights {}.'.format(self.weight_feature_key))
    weight_tensor = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(
        weight_tensor)
    if self.dtype != weight_tensor.dtype.base_dtype:
      raise ValueError('Bad dtype, expected {}, but got {}.'.format(
          self.dtype, weight_tensor.dtype))
    if not isinstance(weight_tensor, sparse_tensor_lib.SparseTensor):
      # The weight tensor can be a regular Tensor. In this case, sparsify it.
      weight_tensor = _to_sparse_input_and_drop_ignore_values(
          weight_tensor, ignore_value=0.0)
    if not weight_tensor.dtype.is_floating:
      weight_tensor = math_ops.cast(weight_tensor, dtypes.float32)
    return weight_tensor

  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Produces the pair of transformed tensors. Inputs: cache and state manager. Reads the weight
    # feature from the cache and converts it, reads the wrapped column's transformed ids and makes them
    # sparse dropping ignore values. Returns a two-element tuple of sparse ids and sparse weights.
    """Applies weights to tensor generated from `categorical_column`'."""
    weight_tensor = transformation_cache.get(self.weight_feature_key,
                                             state_manager)
    sparse_weight_tensor = self._transform_weight_tensor(weight_tensor)
    sparse_categorical_tensor = _to_sparse_input_and_drop_ignore_values(
        transformation_cache.get(self.categorical_column, state_manager))
    return (sparse_categorical_tensor, sparse_weight_tensor)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) transform. Input: legacy input cache. Converts the weight feature
    # and returns a tuple of the wrapped column's tensor and the sparse weights. Emits a deprecation
    # warning.
    """Applies weights to tensor generated from `categorical_column`'."""
    weight_tensor = inputs.get(self.weight_feature_key)
    weight_tensor = self._transform_weight_tensor(weight_tensor)
    return (inputs.get(self.categorical_column), weight_tensor)

  def get_sparse_tensors(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns an IdWeightPair built from the cached transformed (ids, weights) tuple of this column.
    # Inputs: cache and state manager.
    """See `CategoricalColumn` base class."""
    tensors = transformation_cache.get(self, state_manager)
    return CategoricalColumn.IdWeightPair(tensors[0], tensors[1])

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self,
                          inputs,
                          weight_collections=None,
                          trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) getter. Inputs: legacy input cache; weight collections and trainable
    # flag are ignored. Returns an IdWeightPair from the cached tuple. Emits a deprecation warning.
    del weight_collections
    del trainable
    tensors = inputs.get(self)
    return CategoricalColumn.IdWeightPair(tensors[0], tensors[1])

  @property
  def parents(self):
    # [FUNCTION DOC]
    # Property: list with the wrapped categorical column and the weight feature key.
    """See 'FeatureColumn` base class."""
    return [self.categorical_column, self.weight_feature_key]

  def get_config(self):
    # [FUNCTION DOC]
    # Serialises to a mapping of field names to values, replacing the wrapped column with its own
    # serialized configuration and the dtype with its name. The serialization module is imported lazily
    # to avoid a circular import. Returns the mapping.
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['categorical_column'] = serialize_feature_column(
        self.categorical_column)
    config['dtype'] = self.dtype.name
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # Class method rebuilding the column from a configuration mapping. Inputs: mapping, optional custom
    # objects and optional already-revived columns by name. Validates keys (value error on mismatch),
    # converts lists to tuples, deserializes the wrapped column, converts the dtype name to a dtype and
    # returns a new column.
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['categorical_column'] = deserialize_feature_column(
        config['categorical_column'], custom_objects, columns_by_name)
    kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
    return cls(**kwargs)


@serialization.register_feature_column
class CrossedColumn(
    CategoricalColumn,
    fc_old._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('CrossedColumn',
                           ('keys', 'hash_bucket_size', 'hash_key'))):
  # [CLASS DOC]
  # Immutable named tuple (keys, hash_bucket_size, hash_key) produced by crossed_column. Combines
  # several string features and categorical columns (possibly nested crosses) into one hashed cross
  # feature. Registered for serialization.
  """See `crossed_column`."""

  @property
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # Internal property: true only if every leaf-level key (nested crosses expanded) is either a plain
    # string or a feature column that is itself v2.
    for key in _collect_leaf_level_keys(self):
      if isinstance(key, six.string_types):
        continue
      if not isinstance(key, fc_types.FeatureColumn):
        return False
      if not key._is_v2_column:  # pylint: disable=protected-access
        return False
    return True

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: the sorted names of all leaf-level keys (strings as they are, columns by their name)
    # joined with a separator, so the name is independent of key order and nesting.
    """See `FeatureColumn` base class."""
    feature_names = []
    for key in _collect_leaf_level_keys(self):
      if isinstance(key, (fc_types.FeatureColumn, fc_old._FeatureColumn)):  # pylint: disable=protected-access
        feature_names.append(key.name)
      else:  # key must be a string
        feature_names.append(key)
    return '_X_'.join(sorted(feature_names))

  @property
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # Property: merged parsing specification of all keys. Columns contribute their own (v2 or legacy)
    # specification; plain string keys become variable-length string features. Returns the mapping.
    """See `FeatureColumn` base class."""
    config = {}
    for key in self.keys:
      if isinstance(key, fc_types.FeatureColumn):
        config.update(key.parse_example_spec)
      elif isinstance(key, fc_old._FeatureColumn):  # pylint: disable=protected-access
        config.update(key._parse_example_spec)  # pylint: disable=protected-access
      else:  # key must be a string
        config.update({key: parsing_ops.VarLenFeature(dtypes.string)})
    return config

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property equal to parse_example_spec; emits a deprecation warning.
    return self.parse_example_spec

  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Builds the hashed cross. Inputs: cache and state manager. For each leaf key, gets raw feature
    # tensors for strings or the id tensor of categorical columns. Raises a value error when a column
    # supplies weights or when a key has an unsupported type. Combines the tensors with a hashed sparse
    # cross with the configured bucket count and hash key. Returns the sparse cross tensor.
    """Generates a hashed sparse cross from the input tensors."""
    feature_tensors = []
    for key in _collect_leaf_level_keys(self):
      if isinstance(key, six.string_types):
        feature_tensors.append(transformation_cache.get(key, state_manager))
      elif isinstance(key, (fc_old._CategoricalColumn, CategoricalColumn)):  # pylint: disable=protected-access
        ids_and_weights = key.get_sparse_tensors(transformation_cache,
                                                 state_manager)
        if ids_and_weights.weight_tensor is not None:
          raise ValueError(
              'crossed_column does not support weight_tensor, but the given '
              'column populates weight_tensor. '
              'Given column: {}'.format(key.name))
        feature_tensors.append(ids_and_weights.id_tensor)
      else:
        raise ValueError('Unsupported column type. Given: {}'.format(key))
    return sparse_ops.sparse_cross_hashed(
        inputs=feature_tensors,
        num_buckets=self.hash_bucket_size,
        hash_key=self.hash_key)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) transform. Input: legacy input cache. Same as transform_feature but
    # uses the legacy cache and legacy sparse getters; raises value errors for weighted or unsupported
    # keys. Emits a deprecation warning.
    """Generates a hashed sparse cross from the input tensors."""
    feature_tensors = []
    for key in _collect_leaf_level_keys(self):
      if isinstance(key, six.string_types):
        feature_tensors.append(inputs.get(key))
      elif isinstance(key, (CategoricalColumn, fc_old._CategoricalColumn)):  # pylint: disable=protected-access
        ids_and_weights = key._get_sparse_tensors(inputs)  # pylint: disable=protected-access
        if ids_and_weights.weight_tensor is not None:
          raise ValueError(
              'crossed_column does not support weight_tensor, but the given '
              'column populates weight_tensor. '
              'Given column: {}'.format(key.name))
        feature_tensors.append(ids_and_weights.id_tensor)
      else:
        raise ValueError('Unsupported column type. Given: {}'.format(key))
    return sparse_ops.sparse_cross_hashed(
        inputs=feature_tensors,
        num_buckets=self.hash_bucket_size,
        hash_key=self.hash_key)

  @property
  def num_buckets(self):
    # [FUNCTION DOC]
    # Property: number of ids, equal to the hash bucket size.
    """Returns number of buckets in this sparse feature."""
    return self.hash_bucket_size

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property equal to num_buckets; emits a deprecation warning.
    return self.num_buckets

  def get_sparse_tensors(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns an IdWeightPair with the cached sparse cross of this column and no weights. Inputs: cache
    # and state manager.
    """See `CategoricalColumn` base class."""
    return CategoricalColumn.IdWeightPair(
        transformation_cache.get(self, state_manager), None)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self,
                          inputs,
                          weight_collections=None,
                          trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) getter. Inputs: legacy input cache; weight collections and trainable
    # flag are ignored. Returns an IdWeightPair of the cached cross and nothing. Emits a deprecation
    # warning.
    """See `CategoricalColumn` base class."""
    del weight_collections
    del trainable
    return CategoricalColumn.IdWeightPair(inputs.get(self), None)

  @property
  def parents(self):
    # [FUNCTION DOC]
    # Property: list of the column's keys (strings and columns, not expanded).
    """See 'FeatureColumn` base class."""
    return list(self.keys)

  def get_config(self):
    # [FUNCTION DOC]
    # Serialises to a mapping of field names to values with the keys replaced by a tuple of their
    # serialized forms. The serialization module is imported lazily to avoid a circular import.
    # Returns the mapping.
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['keys'] = tuple([serialize_feature_column(fc) for fc in self.keys])
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # Class method rebuilding the column from a configuration mapping. Inputs: mapping, optional custom
    # objects and optional already-revived columns by name. Validates keys (value error on mismatch),
    # converts lists to tuples, deserializes every key into a tuple and returns a new column.
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['keys'] = tuple([
        deserialize_feature_column(c, custom_objects, columns_by_name)
        for c in config['keys']
    ])
    return cls(**kwargs)


def _collect_leaf_level_keys(cross):
  # [FUNCTION DOC]
  # HELPER: Flattening utility for crossed columns. Input: a crossed column. Walks its keys, recursing
  # into nested crossed columns, and returns a flat list of the strings and categorical columns at the
  # leaves, in order.
  """Collects base keys by expanding all nested crosses.

  Args:
    cross: A `CrossedColumn`.

  Returns:
    A list of strings or `CategoricalColumn` instances.
  """
  leaf_level_keys = []
  for k in cross.keys:
    if isinstance(k, CrossedColumn):
      leaf_level_keys.extend(_collect_leaf_level_keys(k))
    else:
      leaf_level_keys.append(k)
  return leaf_level_keys


def _prune_invalid_ids(sparse_ids, sparse_weights):
  # [FUNCTION DOC]
  # HELPER: Removes entries whose id is negative. Inputs: sparse ids and optional sparse weights.
  # Retains only entries with a non-negative id in both (weights are retained at the same positions).
  # Returns the pair of pruned sparse tensors. Appears unused in this module.
  """Prune invalid IDs (< 0) from the input ids and weights."""
  is_id_valid = math_ops.greater_equal(sparse_ids.values, 0)
  if sparse_weights is not None:
    is_id_valid = math_ops.logical_and(
        is_id_valid,
        array_ops.ones_like(sparse_weights.values, dtype=dtypes.bool))
  sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_id_valid)
  if sparse_weights is not None:
    sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_id_valid)
  return sparse_ids, sparse_weights


def _prune_invalid_weights(sparse_ids, sparse_weights):
  # [FUNCTION DOC]
  # HELPER: Removes entries whose weight is not positive. Inputs: sparse ids and optional sparse
  # weights. If weights are given retains only entries with weight greater than zero in both. Returns
  # the pair (ids unchanged when no weights). Appears unused in this module.
  """Prune invalid weights (< 0) from the input ids and weights."""
  if sparse_weights is not None:
    is_weights_valid = math_ops.greater(sparse_weights.values, 0)
    sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_weights_valid)
    sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_weights_valid)
  return sparse_ids, sparse_weights


@serialization.register_feature_column
class IndicatorColumn(
    DenseColumn,
    SequenceDenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    fc_old._SequenceDenseColumn,  # pylint: disable=protected-access
    collections.namedtuple('IndicatorColumn', ('categorical_column'))):
  # [CLASS DOC]
  # Immutable named tuple (categorical_column) produced by indicator_column. Turns a categorical
  # column into a multi-hot (count) dense vector over its buckets, or a dense weighted vector when the
  # wrapped column is weighted. Works both as a dense and a sequence dense column. Registered for
  # serialization.
  """Represents a one-hot column for use in deep networks.

  Args:
    categorical_column: A `CategoricalColumn` which is created by
      `categorical_column_with_*` function.
  """

  @property
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # Internal property: true when the wrapped categorical column is a v2 feature column that is itself v2.
    return (
        isinstance(self.categorical_column, fc_types.FeatureColumn)
        and self.categorical_column._is_v2_column
    )  # pylint: disable=protected-access

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: the wrapped column's name with an indicator suffix.
    """See `FeatureColumn` base class."""
    return '{}_indicator'.format(self.categorical_column.name)

  def _transform_id_weight_pair(self, id_weight_pair, size):
    # [FUNCTION DOC]
    # HELPER: Converts sparse ids (and optional weights) to the dense indicator tensor. Inputs: an
    # IdWeightPair and the number of buckets. With weights, merges ids and weights into a sparse tensor,
    # slices away the extra missing-id column and scatters it into a dense tensor (duplicate ids sum).
    # Without weights, densifies the ids with a fill of minus one, one-hot encodes to float with the
    # given depth and sums over the second-to-last axis to get counts. Returns the dense tensor.
    id_tensor = id_weight_pair.id_tensor
    weight_tensor = id_weight_pair.weight_tensor

    # If the underlying column is weighted, return the input as a dense tensor.
    if weight_tensor is not None:
      weighted_column = sparse_ops.sparse_merge(
          sp_ids=id_tensor, sp_values=weight_tensor, vocab_size=int(size))
      # Remove (?, -1) index.
      weighted_column = sparse_ops.sparse_slice(weighted_column, [0, 0],
                                                weighted_column.dense_shape)
      # Use scatter_nd to merge duplicated indices if existed,
      # instead of sparse_tensor_to_dense.
      return array_ops.scatter_nd(weighted_column.indices,
                                  weighted_column.values,
                                  weighted_column.dense_shape)

    dense_id_tensor = sparse_ops.sparse_tensor_to_dense(
        id_tensor, default_value=-1)

    # One hot must be float for tf.concat reasons since all other inputs to
    # input_layer are float32.
    one_hot_id_tensor = array_ops.one_hot(
        dense_id_tensor, depth=size, on_value=1.0, off_value=0.0)

    # Reduce to get a multi-hot per example.
    return math_ops.reduce_sum(one_hot_id_tensor, axis=[-2])

  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Produces the dense indicator tensor. Inputs: cache and state manager. Gets the wrapped column's
    # sparse tensors and converts them to the dense multi-hot form using the last entry of the column's
    # variable shape as depth. Returns the dense tensor.
    """Returns dense `Tensor` representing feature.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Transformed feature `Tensor`.

    Raises:
      ValueError: if input rank is not known at graph building time.
    """
    id_weight_pair = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    return self._transform_id_weight_pair(id_weight_pair,
                                          self.variable_shape[-1])

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) transform. Input: legacy input cache. Same as transform_feature but
    # uses the legacy sparse getter and legacy variable shape. Emits a deprecation warning.
    id_weight_pair = self.categorical_column._get_sparse_tensors(inputs)  # pylint: disable=protected-access
    return self._transform_id_weight_pair(id_weight_pair,
                                          self._variable_shape[-1])

  @property
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # Property: parsing specification of the wrapped categorical column.
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property returning the wrapped column's legacy parse specification;
    # emits a deprecation warning.
    return self.categorical_column._parse_example_spec  # pylint: disable=protected-access

  @property
  def variable_shape(self):
    # [FUNCTION DOC]
    # Property: tensor shape with two dimensions, one and the wrapped column's bucket count (uses the
    # legacy bucket property for legacy columns).
    """Returns a `TensorShape` representing the shape of the dense `Tensor`."""
    if isinstance(self.categorical_column, fc_types.FeatureColumn):
      return tensor_shape.TensorShape([1, self.categorical_column.num_buckets])
    else:
      return tensor_shape.TensorShape([1, self.categorical_column._num_buckets])  # pylint: disable=protected-access

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _variable_shape(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property; shape of one by the wrapped column's legacy bucket count.
    # Emits a deprecation warning.
    return tensor_shape.TensorShape([1, self.categorical_column._num_buckets])  # pylint: disable=protected-access

  def get_dense_tensor(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns the dense indicator tensor. Inputs: cache and state manager. Raises a value error when the
    # wrapped column is a sequence categorical column. Otherwise returns this column's cached
    # transformed tensor.
    """Returns dense `Tensor` representing feature.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Dense `Tensor` created within `transform_feature`.

    Raises:
      ValueError: If `categorical_column` is a `SequenceCategoricalColumn`.
    """
    if isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In indicator_column: {}. '
          'categorical_column must not be of type SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    # Feature has been already transformed. Return the intermediate
    # representation created by transform_feature.
    return transformation_cache.get(self, state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) dense getter. Inputs: legacy input cache; weight collections and
    # trainable flag are ignored. Raises a value error for a sequence categorical wrapped column
    # (v2 or legacy). Returns the cached transformed tensor. Emits a deprecation warning.
    del weight_collections
    del trainable
    if isinstance(
        self.categorical_column,
        (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):  # pylint: disable=protected-access
      raise ValueError(
          'In indicator_column: {}. '
          'categorical_column must not be of type _SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    # Feature has been already transformed. Return the intermediate
    # representation created by transform_feature.
    return inputs.get(self)

  def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns the indicator tensor and sequence lengths for sequence input. Inputs: cache and state
    # manager. Raises a value error unless the wrapped column is a sequence categorical column. Takes
    # the cached dense tensor, computes lengths from the wrapped column's sparse ids and returns a
    # TensorSequenceLengthPair.
    """See `SequenceDenseColumn` base class."""
    if not isinstance(self.categorical_column, SequenceCategoricalColumn):
      raise ValueError(
          'In indicator_column: {}. '
          'categorical_column must be of type SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    # Feature has been already transformed. Return the intermediate
    # representation created by transform_feature.
    dense_tensor = transformation_cache.get(self, state_manager)
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sequence_dense_tensor(self,
                                 inputs,
                                 weight_collections=None,
                                 trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) sequence getter. Inputs: legacy input cache; weight collections and
    # trainable flag are deleted since no variables are created. Raises a value error unless the wrapped
    # column is a sequence categorical column (v2 or legacy). Returns a TensorSequenceLengthPair of the
    # cached dense tensor and the lengths from the legacy sparse tensors. Emits a deprecation warning.
    # Do nothing with weight_collections and trainable since no variables are
    # created in this function.
    del weight_collections
    del trainable
    if not isinstance(
        self.categorical_column,
        (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):  # pylint: disable=protected-access
      raise ValueError(
          'In indicator_column: {}. '
          'categorical_column must be of type _SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    # Feature has been already transformed. Return the intermediate
    # representation created by _transform_feature.
    dense_tensor = inputs.get(self)
    sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)  # pylint: disable=protected-access
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  @property
  def parents(self):
    # [FUNCTION DOC]
    # Property: list containing only the wrapped categorical column.
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]

  def get_config(self):
    # [FUNCTION DOC]
    # Serialises to a mapping of field names to values, replacing the wrapped column by its serialized
    # configuration (serialization module imported lazily to avoid a circular import). Returns the mapping.
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['categorical_column'] = serialize_feature_column(
        self.categorical_column)
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # Class method rebuilding the column from a configuration mapping. Inputs: mapping, optional custom
    # objects, optional already-revived columns by name. Validates keys (value error on mismatch),
    # converts lists to tuples, deserializes the wrapped column and returns a new column.
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['categorical_column'] = deserialize_feature_column(
        config['categorical_column'], custom_objects, columns_by_name)
    return cls(**kwargs)


def _verify_static_batch_size_equality(tensors, columns):
  # [FUNCTION DOC]
  # HELPER: Consistency check for batch sizes of input tensors.
  # KNOWN ISSUE: The local variable holding the index of the reference column is misspelled with a
  # leading bath instead of batch; it works because it is used consistently. The function is also not
  # called anywhere in this module.
  # Inputs: an indexable collection of tensors and the
  # matching feature columns (used only for error messages). Compares the statically known first
  # dimension of each tensor with the first one that is known; unknown sizes are skipped. Raises a
  # value error naming both columns and sizes on a mismatch. Returns nothing.
  """Verify equality between static batch sizes.

  Args:
    tensors: iterable of input tensors.
    columns: Corresponding feature columns.

  Raises:
    ValueError: in case of mismatched batch sizes.
  """
  # bath_size is a Dimension object.
  expected_batch_size = None
  for i in range(0, len(tensors)):
    batch_size = tensor_shape.Dimension(
        tensor_shape.dimension_value(tensors[i].shape[0]))
    if batch_size.value is not None:
      if expected_batch_size is None:
        bath_size_column_index = i
        expected_batch_size = batch_size
      elif not expected_batch_size.is_compatible_with(batch_size):
        raise ValueError(
            'Batch size (first dimension) of each feature must be same. '
            'Batch size of columns ({}, {}): ({}, {})'.format(
                columns[bath_size_column_index].name, columns[i].name,
                expected_batch_size, batch_size))


@serialization.register_feature_column
class SequenceCategoricalColumn(
    CategoricalColumn,
    fc_old._SequenceCategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple('SequenceCategoricalColumn',
                           ('categorical_column'))):
  # [CLASS DOC]
  # Immutable named tuple (categorical_column) that marks a categorical column as holding sequence
  # data. Delegates name, parsing and transformation to the wrapped column and reshapes the sparse ids
  # to three dimensions (batch, sequence, features) so embeddings are not combined across the
  # sequence. Registered for serialization.
  """Represents sequences of categorical data."""

  @property
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # Internal property: true when the wrapped categorical column is a v2 feature column that is itself v2.
    return (
        isinstance(self.categorical_column, fc_types.FeatureColumn)
        and self.categorical_column._is_v2_column
    )  # pylint: disable=protected-access

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: the wrapped column's name.
    """See `FeatureColumn` base class."""
    return self.categorical_column.name

  @property
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # Property: parsing specification of the wrapped categorical column.
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _parse_example_spec(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property returning the wrapped column's legacy parse specification;
    # emits a deprecation warning.
    return self.categorical_column._parse_example_spec  # pylint: disable=protected-access

  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Delegates to the wrapped column's transform. Inputs: cache and state manager. Returns its result.
    """See `FeatureColumn` base class."""
    return self.categorical_column.transform_feature(transformation_cache,
                                                     state_manager)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _transform_feature(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) transform delegating to the wrapped column's legacy transform with
    # the legacy input cache. Emits a deprecation warning.
    return self.categorical_column._transform_feature(inputs)  # pylint: disable=protected-access

  @property
  def num_buckets(self):
    # [FUNCTION DOC]
    # Property: number of buckets of the wrapped categorical column.
    """Returns number of buckets in this sparse feature."""
    return self.categorical_column.num_buckets

  @property
  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _num_buckets(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) property returning the wrapped column's legacy bucket count; emits a
    # deprecation warning.
    return self.categorical_column._num_buckets  # pylint: disable=protected-access

  def _get_sparse_tensors_helper(self, sparse_tensors):
    # [FUNCTION DOC]
    # HELPER: Reshapes ids (and weights) to three dimensions. Input: an IdWeightPair. Reads the dynamic
    # shape of the id tensor and builds a target shape of batch size, second dimension, and the product
    # of the remaining dimensions (computed explicitly rather than minus one so zero-length dynamic
    # sequences work). Reshapes ids and, when present, weights. Returns a new IdWeightPair.
    id_tensor = sparse_tensors.id_tensor
    weight_tensor = sparse_tensors.weight_tensor
    # Expands third dimension, if necessary so that embeddings are not
    # combined during embedding lookup. If the tensor is already 3D, leave
    # as-is.
    shape = array_ops.shape(id_tensor)
    # Compute the third dimension explicitly instead of setting it to -1, as
    # that doesn't work for dynamically shaped tensors with 0-length at runtime.
    # This happens for empty sequences.
    target_shape = [shape[0], shape[1], math_ops.reduce_prod(shape[2:])]
    id_tensor = sparse_ops.sparse_reshape(id_tensor, target_shape)
    if weight_tensor is not None:
      weight_tensor = sparse_ops.sparse_reshape(weight_tensor, target_shape)
    return CategoricalColumn.IdWeightPair(id_tensor, weight_tensor)

  def get_sparse_tensors(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Returns the wrapped column's sparse tensors reshaped to rank three. Inputs: cache and state
    # manager. Gets the pair from the wrapped column and applies the reshape helper. Returns an
    # IdWeightPair.
    """Returns an IdWeightPair.

    `IdWeightPair` is a pair of `SparseTensor`s which represents ids and
    weights.

    `IdWeightPair.id_tensor` is typically a `batch_size` x `num_buckets`
    `SparseTensor` of `int64`. `IdWeightPair.weight_tensor` is either a
    `SparseTensor` of `float` or `None` to indicate all weights should be
    taken to be 1. If specified, `weight_tensor` must have exactly the same
    shape and indices as `sp_ids`. Expected `SparseTensor` is same as parsing
    output of a `VarLenFeature` which is a ragged matrix.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.
    """
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    return self._get_sparse_tensors_helper(sparse_tensors)

  @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE,
                          _FEATURE_COLUMN_DEPRECATION)
  def _get_sparse_tensors(self,
                          inputs,
                          weight_collections=None,
                          trainable=None):
    # [FUNCTION DOC]
    # HELPER: Deprecated legacy (v1) getter. Inputs: legacy input cache; weight collections and trainable
    # flag are ignored. Fetches the legacy sparse tensors from the wrapped column and reshapes them.
    # Returns an IdWeightPair. Emits a deprecation warning.
    sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)  # pylint: disable=protected-access
    return self._get_sparse_tensors_helper(sparse_tensors)

  @property
  def parents(self):
    # [FUNCTION DOC]
    # Property: list containing only the wrapped categorical column.
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]

  def get_config(self):
    # [FUNCTION DOC]
    # Serialises to a mapping of field names to values, replacing the wrapped column with its serialized
    # configuration (serialization module imported lazily). Returns the mapping.
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top
    config = dict(zip(self._fields, self))
    config['categorical_column'] = serialize_feature_column(
        self.categorical_column)
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # Class method rebuilding the column from a configuration mapping. Inputs: mapping, optional custom
    # objects, optional already-revived columns by name. Validates keys (value error on mismatch),
    # converts lists to tuples, deserializes the wrapped column and returns a new column.
    """See 'FeatureColumn` base class."""
    from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top
    _check_config_keys(config, cls._fields)
    kwargs = _standardize_and_copy_config(config)
    kwargs['categorical_column'] = deserialize_feature_column(
        config['categorical_column'], custom_objects, columns_by_name)
    return cls(**kwargs)


def _check_config_keys(config, expected_keys):
  # [FUNCTION DOC]
  # HELPER: Validation used by every from_config. Inputs: a configuration mapping and the expected
  # key names. Raises a value error when the set of keys differs from the expected set (missing or
  # extra keys). Returns nothing.
  """Checks that a config has all expected_keys."""
  if set(config.keys()) != set(expected_keys):
    raise ValueError('Invalid config: {}, expected keys: {}'.format(
        config, expected_keys))


def _standardize_and_copy_config(config):
  # [FUNCTION DOC]
  # HELPER: Used by from_config methods. Input: a configuration mapping. Returns a shallow copy in
  # which every list value is converted to a tuple (serialization turns tuples into lists, which would
  # make shapes unhashable). The input is not modified.
  """Returns a shallow copy of config with lists turned to tuples.

  Keras serialization uses nest to listify everything.
  This causes problems with the NumericColumn shape, which becomes
  unhashable. We could try to solve this on the Keras side, but that
  would require lots of tracking to avoid changing existing behavior.
  Instead, we ensure here that we revive correctly.

  Args:
    config: dict that will be used to revive a Feature Column

  Returns:
    Shallow copy of config with lists turned to tuples.
  """
  kwargs = config.copy()
  for k, v in kwargs.items():
    if isinstance(v, list):
      kwargs[k] = tuple(v)

  return kwargs


def _sanitize_column_name_for_variable_scope(name):
  # [FUNCTION DOC]
  # HELPER: Used when naming scopes in the transform helper. Input: a user-provided feature name.
  # Returns the name with every character other than letters, digits, underscore, period and hyphen
  # replaced by an underscore.
  """Sanitizes user-provided feature names for use as variable scopes."""
  invalid_char = re.compile('[^A-Za-z0-9_.\\-]')
  return invalid_char.sub('_', name)
