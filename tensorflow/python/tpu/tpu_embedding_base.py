# [MODULE DOC]
# HELPER: shared base layer for the TPU embedding mid level APIs (serving and training variants).
# Defines TPUEmbeddingBase, which validates the feature configuration and the table configuration
# (unique table names, optimizer type checks) and provides the lazy variable-building and call
# protocol. Concrete subclasses supply the variable creation and the actual embedding lookup.
# No import-time side effects beyond importing the TensorFlow modules it depends on.
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
"""Base Class for TPU Embeddings Mid level APIs."""

import functools
from typing import Any, Dict, Iterable, Optional, Union, Text

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest


class TPUEmbeddingBase(autotrackable.AutoTrackable):
  # [CLASS DOC]
  # Abstract base class for TPU embedding mid level APIs. It holds the user feature configuration,
  # the per-feature output shapes, the ordered list of unique table configurations (ordered by
  # first use in the features) and a built flag. Variables are created lazily by build, and calling
  # the object performs an embedding lookup. Subclasses must implement the embedding_tables
  # property, the variable and slot creation, and the lookup. Inherits checkpoint tracking from
  # the auto-trackable base.
  """The TPUEmbedding Base class.

  This class only contains the basic logic to check the feature config and table
  config for the tpu embedding mid level APIs.
  """

  def __init__(
      self,
      feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable],  # pylint:disable=g-bare-generic
      optimizer: Optional[tpu_embedding_v2_utils._Optimizer] = None):  # pylint:disable=protected-access
    # [FUNCTION DOC]
    # Constructor. Takes feature_config (a single feature configuration or a nested structure of
    # them) and an optional default optimizer. Records the feature config, collects each feature's
    # output shape, and builds the list of distinct tables in order of first appearance. For each
    # table with no optimizer the default optimizer is assigned (mutating the table config); a
    # table name of nothing gets an automatic name of the form table_<index>. Raises a value error
    # if a table's optimizer is not an instance of the supported optimizer classes, or if two
    # tables share a name. Marks the object as not yet built. Returns nothing.
    """Creates the TPUEmbeddingBase object."""
    self._feature_config = feature_config
    self._output_shapes = []
    for feature in nest.flatten(feature_config):
      self._output_shapes.append(feature.output_shape)
    # Set table order here to the order of the first occurrence of the table in
    # a feature provided by the user. The order of this struct must be fixed
    # to provide the user with deterministic behavior over multiple
    # instantiations.
    self._table_config = []
    for feature in nest.flatten(feature_config):
      if feature.table not in self._table_config:
        self._table_config.append(feature.table)

    # Ensure tables have unique names. Also error check the optimizer as we
    # specifically don't do that in the TableConfig class to allow high level
    # APIs that are built on this to use strings/other classes to represent
    # optimizers (before they are passed to this class).
    table_names = []
    for i, table in enumerate(self._table_config):
      if table.optimizer is None:
        # TODO(bfontain) Should we allow some sort of optimizer merging here?
        table.optimizer = optimizer
      if (table.optimizer is not None and
          not isinstance(table.optimizer, tpu_embedding_v2_utils._Optimizer)):  # pylint: disable=protected-access
        raise ValueError("{} is an unsupported optimizer class. Please pass an "
                         "instance of one of the optimizer classes under "
                         "tf.tpu.experimental.embedding.".format(
                             type(table.optimizer)))
      if table.name is None:
        table.name = "table_{}".format(i)
      if table.name in table_names:
        raise ValueError("Tables must have a unique name. "
                         f"Multiple tables with name {table.name} found.")
      table_names.append(table.name)

    self._built = False

  @property
  def embedding_tables(self):
    # [FUNCTION DOC]
    # Abstract property: subclasses should return a mapping from table configuration to its
    # embedding table variable(s). This base version always raises a not-implemented error.
    """Returns a dict of embedding tables, keyed by `TableConfig`."""
    raise NotImplementedError

  def _create_variables(self, table: tpu_embedding_v2_utils.TableConfig,
                        trainable: bool) -> Dict[Text, tf_variables.Variable]:
    # [FUNCTION DOC]
    # HELPER: used by subclasses when building the variables for one embedding table.
    # Creates the main parameter variable for the given table configuration plus any optimizer slot
    # variables. Inputs: table (the table config, supplying name, vocabulary size, dimension,
    # initializer and optimizer) and trainable (whether the parameters are trainable; slots never
    # are). Variables are created through the checkpoint-aware custom getter so restores can happen
    # before creation, always as 32-bit floats with shape vocabulary size by dimension. Returns a
    # mapping from slot name to variable, with the main variable stored under the key
    # "parameters". Side effect: registers the variables as dependencies of this object.
    """Create all variables including table variables and slot variables."""
    variable_shape = (table.vocabulary_size, table.dim)

    def getter(name, shape, dtype, initializer, trainable):
      # [FUNCTION DOC]
      # HELPER: custom variable getter for the checkpoint-aware variable creation.
      # Creates a variable with the given name, dtype and trainable flag, using the table's full
      # shape captured from the enclosing scope; the shape argument is deleted and ignored because
      # the tracking machinery sometimes clears it. The initializer is wrapped to produce the
      # table shape lazily. Returns the new variable.
      del shape
      # _add_variable_with_custom_getter clears the shape sometimes, so we
      # take the global shape from outside the getter.
      initial_value = functools.partial(
          initializer, variable_shape, dtype=dtype)
      return tf_variables.Variable(
          name=name,
          initial_value=initial_value,
          shape=variable_shape,
          dtype=dtype,
          trainable=trainable)

    def variable_creator(name, initializer, trainable=True):
      # [FUNCTION DOC]
      # HELPER: creates one tracked variable (table or slot) via the custom getter.
      # Inputs: name, initializer, and trainable (default true). Always uses the table shape and a
      # 32-bit float type. Returns the variable created and tracked on this object, allowing
      # restore-before-create from checkpoints.
      # Use add_variable_with_custom_getter here so that we take advantage of
      # the checkpoint loading to allow restore before the variables get
      # created which avoids double initialization.
      return self._add_variable_with_custom_getter(
          name=name,
          initializer=initializer,
          shape=variable_shape,
          dtype=dtypes.float32,
          getter=getter,
          trainable=trainable)

    parameters = variable_creator(
        table.name, table.initializer, trainable=trainable)

    def slot_creator(name, initializer):
      # [FUNCTION DOC]
      # HELPER: callback handed to the optimizer so it can create its slot variables.
      # Creates a non-trainable variable named after the table name, a slash and the slot name,
      # with the given initializer. Returns the variable.
      return variable_creator(table.name + "/" + name, initializer, False)

    if table.optimizer is not None:
      slot_vars = table.optimizer._create_slots(parameters, slot_creator)  # pylint: disable=protected-access
    else:
      slot_vars = {}
    slot_vars["parameters"] = parameters
    return slot_vars

  def _create_variables_and_slots(self):
    # [FUNCTION DOC]
    # Abstract hook called by build: subclasses create all table and slot variables and return
    # them. This base version always raises a not-implemented error.
    """Create variables and slots variables for TPU embeddings."""
    raise NotImplementedError

  def build(self):
    # [FUNCTION DOC]
    # Idempotent lazy initialization. If already built it returns immediately; otherwise it calls
    # the subclass hook to create the variables, stores the result on the object and marks it
    # built. Takes no inputs and returns nothing; may propagate the hook's errors.
    """Create variables and slots variables for TPU embeddings."""
    if self._built:
      return
    self._variables = self._create_variables_and_slots()
    self._built = True

  def __call__(self, features: Any, weights: Optional[Any] = None) -> Any:
    # [FUNCTION DOC]
    # Makes the object callable as an embedding lookup. Builds the variables first if that has
    # not happened yet, then delegates to embedding_lookup. Inputs: features (structure of input
    # tensors matching the feature config) and optional weights (matching structure, or nothing).
    # Returns the structure of embedded activations produced by embedding_lookup.
    """Call the mid level api to do embedding lookup."""
    if not self._built:
      self.build()
    return self.embedding_lookup(features, weights)

  def embedding_lookup(self,
                       features: Any,
                       weights: Optional[Any] = None) -> Any:
    # [FUNCTION DOC]
    # Abstract lookup: subclasses take features and optional weights and return the embedded
    # activations. This base version always raises a not-implemented error.
    """Lookup the embedding table using the input features."""
    raise NotImplementedError
