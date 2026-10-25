# [MODULE DOC]
# Utility for editing the method names of signature definitions inside an existing SavedModel on
# disk. Exposes the updater class (exported in the v1 API) that parses the SavedModel, modifies
# signature method names in memory, and rewrites the file in text or binary form. Import-time
# effect: registers the API export.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""SignatureDef method name utility functions.

Utility functions for manipulating signature_def.method_names.
"""

from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl as loader
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export


# TODO(jdchung): Consider integrated this into the saved_model_cli so that users
# could do this from the command line directly.
@tf_export(v1=["saved_model.signature_def_utils.MethodNameUpdater"])
class MethodNameUpdater(object):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring's note names the alias builder.MethodNameUpdater, but the class is
  # exported as signature_def_utils.MethodNameUpdater in the v1 API.
  # Utility that parses a SavedModel from an export directory, lets callers replace the method
  # name of signature definitions (across all MetaGraphs or the one matching given tags), and
  # writes the modified SavedModel back (in place or to a new directory) in the same format
  # (text or binary) as the input. Holds the export directory and the parsed SavedModel message.
  """Updates the method name(s) of the SavedModel stored in the given path.

  The `MethodNameUpdater` class provides the functionality to update the method
  name field in the signature_defs of the given SavedModel. For example, it
  can be used to replace the `predict` `method_name` to `regress`.

  Typical usages of the `MethodNameUpdater`
  ```python
  ...
  updater = tf.compat.v1.saved_model.signature_def_utils.MethodNameUpdater(
      export_dir)
  # Update all signature_defs with key "foo" in all meta graph defs.
  updater.replace_method_name(signature_key="foo", method_name="regress")
  # Update a single signature_def with key "bar" in the meta graph def with
  # tags ["serve"]
  updater.replace_method_name(signature_key="bar", method_name="classify",
                              tags="serve")
  updater.save(new_export_dir)
  ```

  Note: This function will only be available through the v1 compatibility
  library as tf.compat.v1.saved_model.builder.MethodNameUpdater.
  """

  def __init__(self, export_dir):
    # [FUNCTION DOC]
    # Constructor. Input: export_dir, the directory containing the SavedModel file. Remembers the
    # directory and parses the SavedModel (text or binary form) into memory. Raises an I/O style
    # error (as raised by the parser) if the file is missing or cannot be parsed. Returns nothing.
    """Creates an MethodNameUpdater object.

    Args:
      export_dir: Directory containing the SavedModel files.

    Raises:
      IOError: If the saved model file does not exist, or cannot be successfully
      parsed.
    """
    self._export_dir = export_dir
    self._saved_model = loader.parse_saved_model(export_dir)

  def replace_method_name(self, signature_key, method_name, tags=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says tags may be a sequence of tags, but only a list is kept as
    # is; any other value (including a tuple or set) is wrapped in a one-element list, so such
    # sequences never match. Also, with tags none, a MetaGraph lacking the key raises instead of
    # being skipped, even though other MetaGraphs may have it; earlier MetaGraphs may already
    # have been modified when the error is raised.
    # Sets the method name of the signature definition with the given key in the in-memory
    # SavedModel: in every MetaGraph if tags is nothing, otherwise in MetaGraphs whose tag set
    # equals the given tags (a single string tag is accepted). Raises a value error if the key or
    # method name is empty, if a selected MetaGraph lacks the key, or if no MetaGraph matched.
    # Mutates the parsed SavedModel; nothing is written to disk. Returns nothing.
    """Replaces the method_name in the specified signature_def.

    This will match and replace multiple sig defs iff tags is None (i.e when
    multiple `MetaGraph`s have a signature_def with the same key).
    If tags is not None, this will only replace a single signature_def in the
    `MetaGraph` with matching tags.

    Args:
      signature_key: Key of the signature_def to be updated.
      method_name: new method_name to replace the existing one.
      tags: A tag or sequence of tags identifying the `MetaGraph` to update. If
          None, all meta graphs will be updated.
    Raises:
      ValueError: if signature_key or method_name are not defined or
          if no metagraphs were found with the associated tags or
          if no meta graph has a signature_def that matches signature_key.
    """
    if not signature_key:
      raise ValueError("`signature_key` must be defined.")
    if not method_name:
      raise ValueError("`method_name` must be defined.")

    if (tags is not None and not isinstance(tags, list)):
      tags = [tags]
    found_match = False
    for meta_graph_def in self._saved_model.meta_graphs:
      if tags is None or set(tags) == set(meta_graph_def.meta_info_def.tags):
        if signature_key not in meta_graph_def.signature_def:
          raise ValueError(
              f"MetaGraphDef associated with tags {tags} "
              f"does not have a signature_def with key: '{signature_key}'. "
              "This means either you specified the wrong signature key or "
              "forgot to put the signature_def with the corresponding key in "
              "your SavedModel.")
        meta_graph_def.signature_def[signature_key].method_name = method_name
        found_match = True

    if not found_match:
      raise ValueError(
          f"MetaGraphDef associated with tags {tags} could not be found in "
          "SavedModel. This means either you specified invalid tags or your "
          "SavedModel does not have a MetaGraphDef with the specified tags.")

  def save(self, new_export_dir=None):
    # [FUNCTION DOC]
    # Writes the (possibly modified) SavedModel. Input: new_export_dir, the destination directory;
    # if empty or nothing, the original export directory is overwritten. Output format follows
    # the input: if the original directory had the text-format file, the text file is written,
    # otherwise the binary file is written with deterministic serialization. The destination
    # directory is not created and must already exist. Logs the written path. Raises an
    # operation error from file writing on failure. Returns nothing.
    """Saves the updated `SavedModel`.

    Args:
      new_export_dir: Path where the updated `SavedModel` will be saved. If
          None, the input `SavedModel` will be overriden with the updates.

    Raises:
      errors.OpError: If there are errors during the file save operation.
    """

    is_input_text_proto = file_io.file_exists(
        file_io.join(
            compat.as_bytes(self._export_dir),
            compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT)))
    if not new_export_dir:
      new_export_dir = self._export_dir

    if is_input_text_proto:
      # TODO(jdchung): Add a util for the path creation below.
      path = file_io.join(
          compat.as_bytes(new_export_dir),
          compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
      file_io.write_string_to_file(path, str(self._saved_model))
    else:
      path = file_io.join(
          compat.as_bytes(new_export_dir),
          compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
      file_io.write_string_to_file(
          path, self._saved_model.SerializeToString(deterministic=True))
    tf_logging.info("SavedModel written to: %s", compat.as_text(path))
