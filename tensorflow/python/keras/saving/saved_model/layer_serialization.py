# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# Savers that serialize Keras layers into SavedModel: LayerSavedModelSaver (generic layers),
# InputLayerSavedModelSaver, RNNSavedModelSaver and IndexLookupLayerSavedModelSaver, plus the
# get_serialized helper. They build the JSON metadata (name, dtype policy, input spec, config)
# and the extra objects and functions that get stored, using the serialized attributes
# machinery and the shared per-export cache. Subclasses of the base saver module's abstract
# class; used by the model and network savers as well. No import-time side effects.
"""Classes and functions implementing Layer SavedModel serialization."""

from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import base_serialization
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest


class LayerSavedModelSaver(base_serialization.SavedModelSaver):
  # [CLASS DOC]
  # Saver for ordinary Keras layers. Produces the layer identifier, the metadata mapping
  # (name, trainable flag, dtype policy, stateful flag, input spec, regularizer, config,
  # build shape) and the extra objects and functions, cached per layer for one export.
  # Subclassed for RNN layers, index lookup layers, models and networks.
  """Implements Layer SavedModel serialization."""

  @property
  def object_identifier(self):
    # [FUNCTION DOC]
    # Returns the constant string marking a generic Keras layer in the SavedModel.
    return constants.LAYER_IDENTIFIER

  @property
  def python_properties(self):
    # [FUNCTION DOC]
    # Returns the mapping of python properties saved as metadata, by delegating to the
    # internal builder. No arguments, no side effects.
    # TODO(kathywu): Add python property validator
    return self._python_properties_internal()

  def _python_properties_internal(self):
    # [FUNCTION DOC]
    # HELPER: shared builder for python_properties, reused by subclasses.
    # Builds a mapping with the layer name, trainable flag, whether call takes a training
    # argument, serialized dtype policy, batch input shape (nothing if absent), stateful
    # flag and whether it must be restored from config. Merges in the serialized config via
    # get_serialized, then adds the serialized input spec (when set), the serialized
    # activity regularizer (only if it has a config method) and the build input shape (when
    # known). Returns the mapping; does not mutate the layer.
    """Returns dictionary of all python properties."""
    # TODO(kathywu): Add support for metrics serialization.
    # TODO(kathywu): Synchronize with the keras spec (go/keras-json-spec) once
    # the python config serialization has caught up.
    metadata = dict(
        name=self.obj.name,
        trainable=self.obj.trainable,
        expects_training_arg=self.obj._expects_training_arg,  # pylint: disable=protected-access
        dtype=policy.serialize(self.obj._dtype_policy),  # pylint: disable=protected-access
        batch_input_shape=getattr(self.obj, '_batch_input_shape', None),
        stateful=self.obj.stateful,
        must_restore_from_config=self.obj._must_restore_from_config,  # pylint: disable=protected-access
    )

    metadata.update(get_serialized(self.obj))
    if self.obj.input_spec is not None:
      # Layer's input_spec has already been type-checked in the property setter.
      metadata['input_spec'] = nest.map_structure(
          lambda x: generic_utils.serialize_keras_object(x) if x else None,
          self.obj.input_spec)
    if (self.obj.activity_regularizer is not None and
        hasattr(self.obj.activity_regularizer, 'get_config')):
      metadata['activity_regularizer'] = generic_utils.serialize_keras_object(
          self.obj.activity_regularizer)
    if self.obj._build_input_shape is not None:  # pylint: disable=protected-access
      metadata['build_input_shape'] = self.obj._build_input_shape  # pylint: disable=protected-access
    return metadata

  def objects_to_serialize(self, serialization_cache):
    # [FUNCTION DOC]
    # Returns the mapping of extra trackable objects to save for the layer, taken from the
    # cached serialized attributes. Input serialization_cache is the per-export cache; it may
    # be populated as a side effect.
    return (self._get_serialized_attributes(
        serialization_cache).objects_to_serialize)

  def functions_to_serialize(self, serialization_cache):
    # [FUNCTION DOC]
    # Returns the mapping of extra functions to save for the layer, taken from the cached
    # serialized attributes. Input serialization_cache is the per-export cache; it may be
    # populated as a side effect.
    return (self._get_serialized_attributes(
        serialization_cache).functions_to_serialize)

  def _get_serialized_attributes(self, serialization_cache):
    # [FUNCTION DOC]
    # HELPER: cache-aware provider for objects_to_serialize and functions_to_serialize.
    # Looks up (creating on demand, in the cache under the Keras cache key) the serialized
    # attributes container for this layer; the new empty container is stored in the cache
    # before it is filled, which guards against recursion. If the layer should skip
    # serialization or must be restored from config, the empty container is returned.
    # Otherwise fills and validates it with the objects and functions from the internal
    # builder. Mutates the cache. Validation errors propagate.
    """Generates or retrieves serialized attributes from cache."""
    keras_cache = serialization_cache.setdefault(constants.KERAS_CACHE_KEY, {})
    if self.obj in keras_cache:
      return keras_cache[self.obj]

    serialized_attr = keras_cache[self.obj] = (
        serialized_attributes.SerializedAttributes.new(self.obj))

    if (save_impl.should_skip_serialization(self.obj) or
        self.obj._must_restore_from_config):  # pylint: disable=protected-access
      return serialized_attr

    object_dict, function_dict = self._get_serialized_attributes_internal(
        serialization_cache)

    serialized_attr.set_and_validate_objects(object_dict)
    serialized_attr.set_and_validate_functions(function_dict)
    return serialized_attr

  def _get_serialized_attributes_internal(self, serialization_cache):
    # [FUNCTION DOC]
    # HELPER: builds the raw objects and functions mappings for a layer.
    # Wraps the layer's objects (such as variables and losses) and functions (call and
    # related functions) through the save implementation helpers, and adds a default save
    # signature entry set to nothing, since the attribute validator requires the key.
    # Returns a pair: objects mapping and functions mapping.
    # KNOWN ISSUE: the docstring says it returns a dictionary, but it returns a tuple of two.
    """Returns dictionary of serialized attributes."""
    objects = save_impl.wrap_layer_objects(self.obj, serialization_cache)
    functions = save_impl.wrap_layer_functions(self.obj, serialization_cache)
    # Attribute validator requires that the default save signature is added to
    # function dict, even if the value is None.
    functions['_default_save_signature'] = None
    return objects, functions


# TODO(kathywu): Move serialization utils (and related utils from
# generic_utils.py) to a separate file.
def get_serialized(obj):
  # [FUNCTION DOC]
  # Returns the Keras serialization (class name and config mapping) of obj, used when the
  # object is revived from its config at load time. Runs inside a context that tolerates
  # failed serialization, so objects that cannot be serialized yield a placeholder result
  # instead of raising.
  with generic_utils.skip_failed_serialization():
    # Store the config dictionary, which may be used when reviving the object.
    # When loading, the program will attempt to revive the object from config,
    # and if that fails, the object will be revived from the SavedModel.
    return generic_utils.serialize_keras_object(obj)


class InputLayerSavedModelSaver(base_serialization.SavedModelSaver):
  # [CLASS DOC]
  # Saver for InputLayer objects. Saves only metadata, with no extra objects or functions.
  """InputLayer serialization."""

  @property
  def object_identifier(self):
    # [FUNCTION DOC]
    # Returns the constant string marking an input layer in the SavedModel.
    return constants.INPUT_LAYER_IDENTIFIER

  @property
  def python_properties(self):
    # [FUNCTION DOC]
    # Returns the metadata mapping for an input layer: class name, name, dtype, sparse and
    # ragged flags, batch input shape and the layer's config mapping.

    return dict(
        class_name=type(self.obj).__name__,
        name=self.obj.name,
        dtype=self.obj.dtype,
        sparse=self.obj.sparse,
        ragged=self.obj.ragged,
        batch_input_shape=self.obj._batch_input_shape,  # pylint: disable=protected-access
        config=self.obj.get_config())

  def objects_to_serialize(self, serialization_cache):
    # [FUNCTION DOC]
    # Returns an empty mapping: an input layer has no extra objects to save. The cache
    # argument is ignored.
    return {}

  def functions_to_serialize(self, serialization_cache):
    # [FUNCTION DOC]
    # Returns an empty mapping: an input layer has no extra functions to save. The cache
    # argument is ignored.
    return {}


class RNNSavedModelSaver(LayerSavedModelSaver):
  # [CLASS DOC]
  # Layer saver specialized for recurrent layers: uses the RNN identifier and additionally
  # saves the layer's state objects as trackable data.
  """RNN layer serialization."""

  @property
  def object_identifier(self):
    # [FUNCTION DOC]
    # Returns the constant string marking an RNN layer in the SavedModel.
    return constants.RNN_LAYER_IDENTIFIER

  def _get_serialized_attributes_internal(self, serialization_cache):
    # [FUNCTION DOC]
    # HELPER: extends the parent builder for recurrent layers.
    # Gets the objects and functions mappings from the parent, wraps the layer's states into
    # a trackable structure (a tuple of non-trackable entries, such as empty or all-null
    # states of a stateless convolutional LSTM, is converted to a list first so it can be
    # wrapped), stores it under the states key of the objects mapping and returns the pair
    # of objects and functions mappings.
    objects, functions = (
        super(RNNSavedModelSaver, self)._get_serialized_attributes_internal(
            serialization_cache))
    states = data_structures.wrap_or_unwrap(self.obj.states)
    # SaveModel require all the objects to be Trackable when saving.
    # If the states is still a tuple after wrap_or_unwrap, it means it doesn't
    # contain any trackable item within it, eg empty tuple or (None, None) for
    # stateless ConvLSTM2D. We convert them to list so that wrap_or_unwrap can
    # make it a Trackable again for saving. When loaded, ConvLSTM2D is
    # able to handle the tuple/list conversion.
    if isinstance(states, tuple):
      states = data_structures.wrap_or_unwrap(list(states))
    objects['states'] = states
    return objects, functions


class IndexLookupLayerSavedModelSaver(LayerSavedModelSaver):
  # [CLASS DOC]
  # Layer saver for index lookup layers (such as vocabulary based preprocessing layers). It
  # differs from the generic saver by dropping large static vocabularies from the metadata.
  """Index lookup layer serialization."""

  @property
  def python_properties(self):
    # [FUNCTION DOC]
    # Returns the generic metadata mapping, but if the config says the layer has a static
    # table, sets the vocabulary entry of the config to nothing so the vocabulary is not
    # stored in the metadata. Mutates the config mapping inside the freshly built metadata.
    # Raises a key error if the metadata has no config entry.
    # TODO(kathywu): Add python property validator
    metadata = self._python_properties_internal()
    if metadata['config'].get('has_static_table', False):
      metadata['config']['vocabulary'] = None
    return metadata
