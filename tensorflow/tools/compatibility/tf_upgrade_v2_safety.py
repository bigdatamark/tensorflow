# [MODULE DOC]
# Safety-mode change spec for the TensorFlow 1.x to 2.0 upgrade script. Exposes one class that
# plugs into the generic AST code upgrader (module ast_edits). Unlike the default spec, it applies
# no API renames or argument rewrites; it only redirects TensorFlow imports to the compat v1
# namespace, carries module deprecation warnings, and warns about symbols moved to Addons. Used by
# the upgrade script's main module when the safety mode is selected. No import-time side effects.
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
"""Upgrader for Python scripts from 1.* to 2.0 TensorFlow using SAFETY mode."""

from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2


class TFAPIChangeSpec(ast_edits.APIChangeSpec):
  # [CLASS DOC]
  # Change spec for safety mode, derived from the generic API change spec base class. Holds empty
  # mappings for keyword renames, symbol renames, function-to-function changes, argument reorders
  # and transformers; a mapping of per-symbol warnings (filled with Addons migration notices); the
  # module deprecation table; the import rename rules (tensorflow to its compat v1 namespace, with
  # excluded prefixes); and the maximum submodule depth. Instantiated without arguments and handed
  # to the code upgrader.
  """List of maps that describe what changed in the API."""

  def __init__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the base class constructor is not called, so any state it sets up is missing.
    # Constructor with no inputs; returns nothing. Sets up empty rename, change, reorder, warning
    # and transformer mappings, and the shared module deprecation table. Then, for every symbol in
    # the Addons symbol mappings, stores a warning-level entry saying manual edit is required
    # because the symbol moved to TensorFlow Addons. Finally sets the import rename rule that maps
    # the top-level package to its compat v1 namespace, excluding contrib, flags, compat (v1 and
    # v2) and google prefixes, and sets the maximum submodule depth to 2.
    self.function_keyword_renames = {}
    self.symbol_renames = {}
    self.change_to_function = {}
    self.function_reorders = {}
    self.function_warnings = {}
    self.function_transformers = {}
    self.module_deprecations = module_deprecations_v2.MODULE_DEPRECATIONS

    ## Inform about the addons mappings
    for symbol, replacement in all_renames_v2.addons_symbol_mappings.items():
      warning = (
          ast_edits.WARNING, (
              "(Manual edit required) `{}` has been migrated to `{}` in "
              "TensorFlow Addons. The API spec may have changed during the "
              "migration. Please see https://github.com/tensorflow/addons "
              "for more info.").format(symbol, replacement))
      self.function_warnings[symbol] = warning

    # List module renames. If changed, please update max_submodule_depth.
    self.import_renames = {
        "tensorflow":
            ast_edits.ImportRename(
                "tensorflow.compat.v1",
                excluded_prefixes=[
                    "tensorflow.contrib", "tensorflow.flags",
                    "tensorflow.compat",
                    "tensorflow.compat.v1", "tensorflow.compat.v2",
                    "tensorflow.google"
                ],
            )
    }
    # Needs to be updated if self.import_renames is changed.
    self.max_submodule_depth = 2
