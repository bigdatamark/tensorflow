# [MODULE DOC]
# Persistent user configuration for the TensorFlow Debugger (tfdbg) command-line interfaces.
# Exposes the CLIConfig class, which holds named properties (graph recursion depth and mouse
# mode by default), stores them as a JSON file in the user's home directory, and lets callers
# read, change and summarize them. Used by the base UI class to implement the "config" command.
# A module-level alias for the rich-line class of the common debugger CLI module is defined at
# import time; no other import-time side effects.
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
"""Configurations for TensorFlow Debugger (TFDBG) command-line interfaces."""
import collections
import json
import os

from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.platform import gfile

RL = debugger_cli_common.RichLine


class CLIConfig(object):
  # [CLASS DOC]
  # Holds the user-facing settings of the tfdbg command-line interfaces as an ordered mapping
  # from property names to values (bool, int or string). Defaults are built in; on creation the
  # values stored in a JSON config file (default: a hidden file in the home directory) override
  # them, and the merged result is written back. Changes via the set method are saved
  # immediately and trigger an optional per-property callback.
  """Client-facing configurations for TFDBG command-line interfaces."""

  _CONFIG_FILE_NAME = ".tfdbg_config"

  _DEFAULT_CONFIG = [
      ("graph_recursion_depth", 20),
      ("mouse_mode", True),
  ]

  def __init__(self, config_file_path=None):
    # [FUNCTION DOC]
    # Constructor. config_file_path (string, optional) is the config file location; when
    # missing or empty, the default file in the user's home directory is used. Starts from the
    # default properties, overlays any values loaded from the file if it exists, then writes
    # the merged configuration back to the file (write errors are silently ignored). Also
    # initializes an empty mapping of per-property set-callbacks. Returns nothing.
    self._config_file_path = (config_file_path or
                              self._default_config_file_path())
    self._config = collections.OrderedDict(self._DEFAULT_CONFIG)
    if gfile.Exists(self._config_file_path):
      config = self._load_from_file()
      for key, value in config.items():
        self._config[key] = value
    self._save_to_file()

    self._set_callbacks = {}

  def get(self, property_name):
    # [FUNCTION DOC]
    # Returns the current value of the named property. property_name is a string.
    # Raises a key error if the property does not exist. No side effects.
    if property_name not in self._config:
      raise KeyError("%s is not a valid property name." % property_name)
    return self._config[property_name]

  def set(self, property_name, property_val):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says only bool, int and str are supported and mentions only
    # ValueError and KeyError; the code also raises a type error for any other existing value
    # type. Because bool is checked before int, bool properties never take the int path.
    # Sets a property, coercing the new value to the type of the existing value: for a bool
    # property, strings are parsed from common true/false words (case-insensitive), other
    # values are converted by truthiness; int properties are converted to int; string
    # properties to string. Raises a key error for unknown names, a value error for an
    # unparsable bool string (and for non-numeric int input), a type error for unsupported
    # property types. Side effects: saves the whole config to the file and, if a callback is
    # registered for the property, calls it with the full config mapping. Returns nothing.
    """Set the value of a property.

    Supports limitd property value types: `bool`, `int` and `str`.

    Args:
      property_name: Name of the property.
      property_val: Value of the property. If the property has `bool` type and
        this argument has `str` type, the `str` value will be parsed as a `bool`

    Raises:
      ValueError: if a `str` property_value fails to be parsed as a `bool`.
      KeyError: if `property_name` is an invalid property name.
    """
    if property_name not in self._config:
      raise KeyError("%s is not a valid property name." % property_name)

    orig_val = self._config[property_name]
    if isinstance(orig_val, bool):
      if isinstance(property_val, str):
        if property_val.lower() in ("1", "true", "t", "yes", "y", "on"):
          property_val = True
        elif property_val.lower() in ("0", "false", "f", "no", "n", "off"):
          property_val = False
        else:
          raise ValueError(
              "Invalid string value for bool type: %s" % property_val)
      else:
        property_val = bool(property_val)
    elif isinstance(orig_val, int):
      property_val = int(property_val)
    elif isinstance(orig_val, str):
      property_val = str(property_val)
    else:
      raise TypeError("Unsupported property type: %s" % type(orig_val))
    self._config[property_name] = property_val
    self._save_to_file()

    # Invoke set-callback.
    if property_name in self._set_callbacks:
      self._set_callbacks[property_name](self._config)

  def set_callback(self, property_name, callback):
    # [FUNCTION DOC]
    # Registers a callback for a property; it is called with the full config mapping after each
    # successful set of that property. Only one callback per property is kept (a new one
    # replaces the old). Raises a key error for unknown property names and a type error if the
    # callback is not callable. Returns nothing.
    """Set a set-callback for given property.

    Args:
      property_name: Name of the property.
      callback: The callback as a `callable` of signature:
          def cbk(config):
        where config is the config after it is set to the new value.
        The callback is invoked each time the set() method is called with the
        matching property_name.

    Raises:
      KeyError: If property_name does not exist.
      TypeError: If `callback` is not callable.
    """
    if property_name not in self._config:
      raise KeyError("%s is not a valid property name." % property_name)
    if not callable(callback):
      raise TypeError("The callback object provided is not callable.")
    self._set_callbacks[property_name] = callback

  def _default_config_file_path(self):
    # [FUNCTION DOC]
    # HELPER: locates the default config file for the constructor.
    # Returns the path of the config file name inside the current user's home directory.
    return os.path.join(os.path.expanduser("~"), self._CONFIG_FILE_NAME)

  def _save_to_file(self):
    # [FUNCTION DOC]
    # HELPER: persistence step used by the constructor and the set method.
    # Writes the whole configuration mapping to the config file as JSON, overwriting it.
    # I/O errors are swallowed silently. Returns nothing.
    try:
      with gfile.Open(self._config_file_path, "w") as config_file:
        json.dump(self._config, config_file)
    except IOError:
      pass

  def summarize(self, highlight=None):
    # [FUNCTION DOC]
    # Builds a rich-text description of the configuration: a bold title, a blank line, then one
    # line per property showing its underlined name and value. highlight (string, optional) is
    # a property name whose name and value are shown in bold. Returns a rich-text-lines
    # object; no side effects.
    """Get a text summary of the config.

    Args:
      highlight: A property name to highlight in the output.

    Returns:
      A `RichTextLines` output.
    """
    lines = [RL("Command-line configuration:", "bold"), RL("")]
    for name, val in self._config.items():
      highlight_attr = "bold" if name == highlight else None
      line = RL("  ")
      line += RL(name, ["underline", highlight_attr])
      line += RL(": ")
      line += RL(str(val), font_attr=highlight_attr)
      lines.append(line)
    return debugger_cli_common.rich_text_lines_from_rich_line_list(lines)

  def _load_from_file(self):
    # [FUNCTION DOC]
    # HELPER: reads the saved configuration for the constructor.
    # Reads the JSON config file and returns an ordered mapping of its entries sorted by key.
    # On I/O errors or corrupt content it returns an empty plain mapping instead of failing.
    try:
      with gfile.Open(self._config_file_path, "r") as config_file:
        config_dict = json.load(config_file)
        config = collections.OrderedDict()
        for key in sorted(config_dict.keys()):
          config[key] = config_dict[key]
        return config
    except (IOError, ValueError):
      # The reading of the config file may fail due to IO issues or file
      # corruption. We do not want tfdbg to error out just because of that.
      return dict()
