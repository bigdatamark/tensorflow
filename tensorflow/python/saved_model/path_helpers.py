# [MODULE DOC]
# HELPER: small helper module that computes (and optionally creates) the standard file and
# directory locations inside an exported SavedModel directory: variables, assets, debug, and the
# saved model protocol buffer files. Used by the SavedModel builder, saver and loader code. All
# paths are built from constants in the saved model constants module; no import-time effects.
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

"""Path helpers utility functions."""

from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat


def get_or_create_variables_dir(export_dir):
  # [FUNCTION DOC]
  # Input: export_dir, the SavedModel directory (text or bytes). Computes the variables
  # sub-directory path, creates it (and missing parents) if absent, and returns the path.
  # Side effect: creates directories on the file system; I/O errors propagate.
  """Return variables sub-directory, or create one if it doesn't exist."""
  variables_dir = get_variables_dir(export_dir)
  file_io.recursive_create_dir(variables_dir)
  return variables_dir


def get_variables_dir(export_dir):
  # [FUNCTION DOC]
  # Input: export_dir, the SavedModel directory. Returns the text path of the variables
  # sub-directory inside it. Pure path computation; touches no files.
  """Return variables sub-directory in the SavedModel."""
  return file_io.join(
      compat.as_text(export_dir), compat.as_text(constants.VARIABLES_DIRECTORY))


def get_variables_path(export_dir):
  # [FUNCTION DOC]
  # Input: export_dir, the SavedModel directory. Returns the text path of the variables file
  # name inside the variables sub-directory; this is the prefix used for checkpoint files.
  # Pure path computation.
  """Return the variables path, used as the prefix for checkpoint files."""
  return file_io.join(
      compat.as_text(get_variables_dir(export_dir)),
      compat.as_text(constants.VARIABLES_FILENAME))


def get_or_create_assets_dir(export_dir):
  # [FUNCTION DOC]
  # Input: export_dir, the SavedModel directory. Computes the assets sub-directory path,
  # creates it (and missing parents) if absent, and returns the path. Side effect: creates
  # directories; I/O errors propagate.
  """Return assets sub-directory, or create one if it doesn't exist."""
  assets_destination_dir = get_assets_dir(export_dir)

  file_io.recursive_create_dir(assets_destination_dir)

  return assets_destination_dir


def get_assets_dir(export_dir):
  # [FUNCTION DOC]
  # Input: export_dir, the SavedModel directory. Returns the text path of the assets
  # sub-directory inside it. Pure path computation.
  """Return path to asset directory in the SavedModel."""
  return file_io.join(
      compat.as_text(export_dir), compat.as_text(constants.ASSETS_DIRECTORY))


def get_or_create_debug_dir(export_dir):
  # [FUNCTION DOC]
  # Input: export_dir, the SavedModel directory. Computes the debug sub-directory path,
  # creates it (and missing parents) if absent, and returns the path. Side effect: creates
  # directories; I/O errors propagate.
  """Returns path to the debug sub-directory, creating if it does not exist."""
  debug_dir = get_debug_dir(export_dir)

  file_io.recursive_create_dir(debug_dir)

  return debug_dir


def get_saved_model_pbtxt_path(export_dir):
  # [FUNCTION DOC]
  # KNOWN ISSUE: unlike the other helpers, which return text paths, this returns a byte string
  # path (the result of joining byte strings); callers must handle bytes.
  # Input: export_dir, the SavedModel directory (text, bytes or path-like). Returns the path of
  # the text-format saved model file inside it. Pure path computation.
  return file_io.join(
      compat.as_bytes(compat.path_to_str(export_dir)),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))


def get_saved_model_pb_path(export_dir):
  # [FUNCTION DOC]
  # KNOWN ISSUE: returns a byte string path, unlike the text paths from the directory helpers.
  # Input: export_dir, the SavedModel directory (text, bytes or path-like). Returns the path of
  # the binary saved model file inside it. Pure path computation.
  return file_io.join(
      compat.as_bytes(compat.path_to_str(export_dir)),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))


def get_debug_dir(export_dir):
  # [FUNCTION DOC]
  # Input: export_dir, the SavedModel directory. Returns the text path of the debug
  # sub-directory inside it. Pure path computation; does not create anything.
  """Returns path to the debug sub-directory in the SavedModel."""
  return file_io.join(
      compat.as_text(export_dir), compat.as_text(constants.DEBUG_DIRECTORY))
