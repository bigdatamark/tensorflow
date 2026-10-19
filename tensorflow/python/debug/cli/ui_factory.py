# [MODULE DOC]
# Factory for the debugger command-line user interface. Defines the list of supported UI types
# and a function that builds a UI object of the requested type, falling back to other available
# types if importing the requested UI module fails. The UI module is imported lazily.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow Debugger (tfdbg) User-Interface Factory."""
import copy


SUPPORTED_UI_TYPES = ["readline"]


def get_ui(ui_type,
           on_ui_exit=None,
           available_ui_types=None,
           config=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: when ui_type is empty or none, no branch matches, so the function returns
  # nothing rather than a UI; and on fallback after an import failure, the on_ui_exit callback
  # and config are not passed on to the fallback UI.
  # Creates a debugger UI object of the requested type (only a line-editing console UI is
  # supported). Parameters: ui_type (string name), on_ui_exit (optional callback run on UI
  # exit), available_ui_types (optional list of allowed names, defaults to a copy of the
  # supported list; it is mutated by removing failed types), config (optional CLI configuration
  # object). Raises a value error if ui_type is non-empty and not available, or when every
  # fallback type has been exhausted after import failures. Returns the UI object.
  """Create a `base_ui.BaseUI` subtype.

  This factory method attempts to fallback to other available ui_types on
  ImportError.

  Args:
    ui_type: (`str`) requested UI type. Currently supported:
      ( readline)
    on_ui_exit: (`Callable`) the callback to be called when the UI exits.
    available_ui_types: (`None` or `list` of `str`) Manually-set available
      ui_types.
    config: An instance of `cli_config.CLIConfig()` carrying user-facing
      configurations.

  Returns:
    A `base_ui.BaseUI` subtype object.

  Raises:
    ValueError: on invalid ui_type or on exhausting or fallback ui_types.
  """
  if available_ui_types is None:
    available_ui_types = copy.deepcopy(SUPPORTED_UI_TYPES)

  if ui_type and (ui_type not in available_ui_types):
    raise ValueError("Invalid ui_type: '%s'" % ui_type)

  try:
    # pylint: disable=g-import-not-at-top
    if ui_type == "readline":
      from tensorflow.python.debug.cli import readline_ui
      return readline_ui.ReadlineUI(on_ui_exit=on_ui_exit, config=config)
    # pylint: enable=g-import-not-at-top
  except ImportError:
    available_ui_types.remove(ui_type)
    if not available_ui_types:
      raise ValueError("Exhausted all fallback ui_types.")
    return get_ui(available_ui_types[0],
                  available_ui_types=available_ui_types)
