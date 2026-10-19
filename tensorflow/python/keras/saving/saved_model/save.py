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
# [MODULE DOC]
# Entry point for saving a Keras model in the SavedModel format. Provides save, which
# handles overwrite confirmation, optional removal of the optimizer, tracing under a fixed
# learning phase and writing the SavedModel plus a separate Keras metadata file, and
# generate_keras_metadata, which builds that metadata message. Import-time side effect:
# creates lazy loaders for the base layer and training modules to avoid circular imports.
"""Keras SavedModel serialization."""

import os

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.protobuf import saved_metadata_pb2
from tensorflow.python.keras.protobuf import versions_pb2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import save as save_lib

# To avoid circular dependencies between keras/engine and keras/saving,
# code in keras/saving must delay imports.

base_layer = LazyLoader(
    "base_layer", globals(),
    "tensorflow.python.keras.engine.base_layer")
training_lib = LazyLoader(
    "training_lib", globals(),
    "tensorflow.python.keras.engine.training")


def save(model, filepath, overwrite, include_optimizer, signatures=None,
         options=None, save_traces=True):
  # [FUNCTION DOC]
  # KNOWN ISSUE: when the optimizer is excluded it is temporarily removed from the model and
  # restored only at the very end without a protective cleanup block, so an exception during
  # saving leaves the model without its optimizer.
  # Saves a Keras model as a SavedModel directory. Inputs: model (Keras model), filepath
  # (string directory), overwrite (boolean; if false and the path exists, asks the user and
  # silently returns without saving when declined), include_optimizer (boolean; if false the
  # optimizer is detached while saving), signatures and options (passed through to the core
  # saver; both optional), save_traces (boolean, default true; if false only configs are
  # stored). If tracing is on and the model has no defined inputs, raises the model input
  # error (a value error). Saves under a learning phase of zero and the trace option scope,
  # then writes the Keras metadata file into the directory. Side effects: file writes, user
  # prompt, temporary mutation of the model's optimizer. Returns nothing.
  """Saves a model as a SavedModel to the filepath.

  Args:
    model: Keras model instance to be saved.
    filepath: String path to save the model.
    overwrite: whether to overwrite the existing filepath.
    include_optimizer: If True, save the model's optimizer state.
    signatures: Signatures to save with the SavedModel. Applicable to the 'tf'
      format only. Please see the `signatures` argument in `tf.saved_model.save`
      for details.
    options: (only applies to SavedModel format) `tf.saved_model.SaveOptions`
      object that specifies options for saving to SavedModel.
    save_traces: (only applies to SavedModel format) When enabled, the
      SavedModel will store the function traces for each layer. This
      can be disabled, so that only the configs of each layer are stored.
      Defaults to `True`. Disabling this will decrease serialization time
      and reduce file size, but it requires that all custom layers/models
      implement a `get_config()` method.

  Raises:
    ValueError: if the model's inputs have not been defined.
  """
  # If file exists and should not be overwritten.
  if not overwrite and os.path.exists(filepath):
    proceed = ask_to_proceed_with_overwrite(filepath)
    if not proceed:
      return

  if save_traces:
    if save_impl.should_skip_serialization(model):
      saving_utils.raise_model_input_error(model)

  if not include_optimizer:
    orig_optimizer = model.optimizer
    model.optimizer = None
    # TODO(b/180760306) Change to del model.optimizer if Layer's __delattr__
    # calls AutoTrackable's __delattr__.
    model._delete_tracking("optimizer")  # pylint: disable=protected-access

  # Trace all functions and signatures with `training=0` instead of using an
  # already-set learning phase placeholder.
  # This is needed for compatibility reasons until learning phase setting
  # is removed from the public apis.
  with K.deprecated_internal_learning_phase_scope(0):
    with utils.keras_option_scope(save_traces):
      saved_nodes, node_paths = save_lib.save_and_return_nodes(
          model, filepath, signatures, options)

    # Save all metadata to a separate file in the SavedModel directory.
    metadata = generate_keras_metadata(saved_nodes, node_paths)

  with gfile.GFile(
      os.path.join(filepath, constants.SAVED_METADATA_PATH), "wb") as w:
    w.write(metadata.SerializeToString(deterministic=True))

  if not include_optimizer:
    model.optimizer = orig_optimizer


def generate_keras_metadata(saved_nodes, node_paths):
  # [FUNCTION DOC]
  # Builds the saved-metadata message for all Keras layers among the saved nodes. Inputs:
  # saved_nodes (ordered list of saved objects; position is the node id) and node_paths
  # (mapping from each object to its list of path references from the root). For each node
  # that is a Keras layer, adds an entry with node id, dotted path (root, or root followed
  # by the reference names), a version record (producer 1, minimum consumer 1, no bad
  # consumers), the object identifier and the tracking metadata. Returns the message.
  # Pure apart from reading object properties.
  """Constructs a KerasMetadata proto with the metadata of each keras object."""
  metadata = saved_metadata_pb2.SavedMetadata()

  for node_id, node in enumerate(saved_nodes):
    if isinstance(node, base_layer.Layer):
      path = node_paths[node]
      if not path:
        node_path = "root"
      else:
        node_path = "root.{}".format(
            ".".join([ref.name for ref in path]))

      metadata.nodes.add(
          node_id=node_id,
          node_path=node_path,
          version=versions_pb2.VersionDef(
              producer=1, min_consumer=1, bad_consumers=[]),
          identifier=node._object_identifier,  # pylint: disable=protected-access
          metadata=node._tracking_metadata)  # pylint: disable=protected-access

  return metadata
