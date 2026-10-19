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
# Defines ModelSavedModelSaver and SequentialSavedModelSaver, which serialize Keras models
# (and Sequential models) into a SavedModel by extending the layer saver with model-specific
# metadata (graph network flag, save spec, model metadata) and a default save signature
# for the root object. No import-time side effects.
"""Classes and functions implementing to Model SavedModel serialization."""

from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.saving.saved_model import save_impl


class ModelSavedModelSaver(layer_serialization.LayerSavedModelSaver):
  # [CLASS DOC]
  # Layer saver specialization for Keras models. Adds model metadata and, for the root
  # model of an export, a default save signature function. Parent of the Sequential and
  # network savers.
  """Model SavedModel serialization."""

  @property
  def object_identifier(self):
    # [FUNCTION DOC]
    # Returns the constant string marking a Keras model in the SavedModel.
    return constants.MODEL_IDENTIFIER

  def _python_properties_internal(self):
    # [FUNCTION DOC]
    # HELPER: extends the layer metadata builder for models.
    # Starts from the layer metadata, removes the stateful entry (it depends on child
    # layers), adds whether the model is a graph network and its save spec (with a fixed
    # batch size), then merges the general model metadata (optimizer included, config not
    # required). Returns the mapping.
    metadata = super(ModelSavedModelSaver, self)._python_properties_internal()
    # Network stateful property is dependent on the child layers.
    metadata.pop('stateful')
    metadata['is_graph_network'] = self.obj._is_graph_network  # pylint: disable=protected-access
    metadata['save_spec'] = self.obj._get_save_spec(dynamic_batch=False)  # pylint: disable=protected-access

    metadata.update(
        saving_utils.model_metadata(
            self.obj, include_optimizer=True, require_config=False))
    return metadata

  def _get_serialized_attributes_internal(self, serialization_cache):
    # [FUNCTION DOC]
    # HELPER: extends the layer builder for models.
    # Input serialization_cache is the per-export cache. If this model is the only entry in
    # the Keras cache (the root object), creates the default save signature function; then
    # obtains objects and functions from the layer builder and stores the signature (or
    # nothing for non-root models) under the default save signature key. Returns the pair of
    # objects and functions mappings.
    default_signature = None

    # Create a default signature function if this is the only object in the
    # cache (i.e. this is the root level object).
    if len(serialization_cache[constants.KERAS_CACHE_KEY]) == 1:
      default_signature = save_impl.default_save_signature(self.obj)

    # Other than the default signature function, all other attributes match with
    # the ones serialized by Layer.
    objects, functions = (
        super(ModelSavedModelSaver, self)._get_serialized_attributes_internal(
            serialization_cache))
    functions['_default_save_signature'] = default_signature
    return objects, functions


class SequentialSavedModelSaver(ModelSavedModelSaver):
  # [CLASS DOC]
  # Model saver for Sequential models; identical to the model saver except that it uses the
  # Sequential identifier.

  @property
  def object_identifier(self):
    # [FUNCTION DOC]
    # Returns the constant string marking a Sequential model in the SavedModel.
    return constants.SEQUENTIAL_IDENTIFIER
