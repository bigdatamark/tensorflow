# [MODULE DOC]
# HELPER: configuration/flag shim used by the tf-numpy test suites (a lightweight imitation of
# JAX's config object). Defines a Config class holding named options with defaults, a namespace
# for attribute-style reads, optional bridging to absl flags, and a boolean environment-variable
# parser. Import-time side effects: creates the global config object (also exposed as flags and
# FLAGS), and defines the options jax_enable_checks (default from the JAX_ENABLE_CHECKS
# environment variable) and tf_numpy_additional_tests (default true).
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Test configurations."""
import os
import sys


def bool_env(varname: str, default: bool) -> bool:
  # [FUNCTION DOC]
  # HELPER: environment parsing for flag defaults.
  # KNOWN ISSUE: the docstring lists the Raises section under Args without a separate heading; it
  # is accurate in substance (a value error is raised for unrecognized text) but the return
  # value is not documented: it returns the boolean.
  # Reads the named environment variable (using the string form of the default when unset),
  # lowercases it, and returns true for y, yes, t, true, on, 1 and false for n, no, f, false,
  # off, 0. Raises a value error naming the variable for any other text.
  """Read an environment variable and interpret it as a boolean.

  True values are (case insensitive): 'y', 'yes', 't', 'true', 'on', and '1';
  false values are 'n', 'no', 'f', 'false', 'off', and '0'.

  Args:
    varname: the name of the variable
    default: the default boolean value
  Raises: ValueError if the environment variable is anything else.
  """
  val = os.getenv(varname, str(default))
  val = val.lower()
  if val in ('y', 'yes', 't', 'true', 'on', '1'):
    return True
  elif val in ('n', 'no', 'f', 'false', 'off', '0'):
    return False
  else:
    raise ValueError(
        'invalid truth value %r for environment %r' % (val, varname)
    )


class Config(object):
  # [CLASS DOC]
  # Registry of named configuration options. Holds a mapping of option names to current values,
  # a mapping of names to (type, positional args, keyword args) metadata used when bridging to
  # absl flags, a FLAGS namespace for attribute-style reads, and a flag telling whether absl is
  # the backing store. Options are declared with the DEFINE methods; one module-level instance
  # is created below.

  def __init__(self):
    # [FUNCTION DOC]
    # Creates empty value and metadata mappings, a FLAGS namespace that reads through this
    # object, and sets absl mode off. Returns nothing.
    self.values = {}
    self.meta = {}
    self.FLAGS = NameSpace(self.read)
    self.use_absl = False

  def update(self, name, val):
    # [FUNCTION DOC]
    # Sets option name to val. In absl mode, assigns the attribute on the absl flag holder
    # (without validating the name). Otherwise checks that the option exists and stores the new
    # value. Raises a generic exception for an unrecognized option name (non-absl mode).
    # Returns nothing.
    if self.use_absl:
      setattr(self.absl_flags.FLAGS, name, val)
    else:
      self.check_exists(name)
      if name not in self.values:
        raise Exception("Unrecognized config option: {}".format(name))
      self.values[name] = val

  def read(self, name):
    # [FUNCTION DOC]
    # Returns the current value of option name: from the absl flag holder in absl mode (may raise
    # an attribute error if undefined), otherwise from the local mapping after checking the name
    # exists (raises a generic exception when unknown).
    if self.use_absl:
      return getattr(self.absl_flags.FLAGS, name)
    else:
      self.check_exists(name)
      return self.values[name]

  def add_option(self, name, default, opt_type, meta_args, meta_kwargs):
    # [FUNCTION DOC]
    # HELPER: shared implementation of the DEFINE methods.
    # Registers a new option: name, default value, option type (bool, int, str or the enum
    # marker), and the positional and keyword arguments to later pass to the absl definer.
    # Raises a generic exception if the name is already defined. Stores the default as the
    # current value and the metadata. Returns nothing.
    if name in self.values:
      raise Exception("Config option {} already defined".format(name))
    self.values[name] = default
    self.meta[name] = (opt_type, meta_args, meta_kwargs)

  def check_exists(self, name):
    # [FUNCTION DOC]
    # HELPER: validation for update and read.
    # Raises a generic exception if no option called name has been defined; otherwise returns
    # nothing.
    if name not in self.values:
      raise Exception("Unrecognized config option: {}".format(name))

  def DEFINE_bool(self, name, default, *args, **kwargs):
    # [FUNCTION DOC]
    # Declares a boolean option with a default; extra arguments (such as help text) are saved for
    # absl. Raises a generic exception if already defined. Returns nothing.
    self.add_option(name, default, bool, args, kwargs)

  def DEFINE_integer(self, name, default, *args, **kwargs):
    # [FUNCTION DOC]
    # Declares an integer option with a default; extra arguments are saved for absl. Raises a
    # generic exception if already defined. Returns nothing.
    self.add_option(name, default, int, args, kwargs)

  def DEFINE_string(self, name, default, *args, **kwargs):
    # [FUNCTION DOC]
    # Declares a string option with a default; extra arguments are saved for absl. Raises a
    # generic exception if already defined. Returns nothing.
    self.add_option(name, default, str, args, kwargs)

  def DEFINE_enum(self, name, default, *args, **kwargs):
    # [FUNCTION DOC]
    # Declares an enumerated option with a default; extra arguments (including the allowed
    # values) are saved for absl. Raises a generic exception if already defined. Returns nothing.
    self.add_option(name, default, 'enum', args, kwargs)

  def config_with_absl(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: imports absl.flags under an unused alias and relies on this being called
    # before the application runs; calling it twice redefines the absl flags and fails.
    # Switches to absl mode: defines one absl flag for each registered option using its stored
    # type, default and metadata, remembers the absl flags module, and registers a callback so
    # that after absl initialization the parsed values are copied back via complete_absl_config.
    # Returns nothing.
    # Run this before calling `app.run(main)` etc
    import absl.flags as absl_FLAGS
    from absl import app, flags as absl_flags

    self.use_absl = True
    self.absl_flags = absl_flags
    absl_defs = { bool: absl_flags.DEFINE_bool,
                  int:  absl_flags.DEFINE_integer,
                  str:  absl_flags.DEFINE_string,
                  'enum': absl_flags.DEFINE_enum }

    for name, val in self.values.items():
      flag_type, meta_args, meta_kwargs = self.meta[name]
      absl_defs[flag_type](name, val, *meta_args, **meta_kwargs)

    app.call_after_init(lambda: self.complete_absl_config(absl_flags))

  def complete_absl_config(self, absl_flags):
    # [FUNCTION DOC]
    # Copies the value of every registered option from the given absl flags module back into
    # this config via update. Returns nothing; may raise an attribute error if a flag is missing.
    for name, _ in self.values.items():
      self.update(name, getattr(absl_flags.FLAGS, name))

  def parse_flags_with_absl(self):
    # [FUNCTION DOC]
    # One-time setup: if not yet done (module-level guard), enables absl mode, parses the
    # process command-line arguments ignoring unknown flags, copies the parsed values into this
    # config, and sets the guard. Does nothing on later calls. Side effects: defines absl flags,
    # reads the program arguments, mutates the module-level guard.
    global already_configured_with_absl
    if not already_configured_with_absl:
      import absl.flags
      self.config_with_absl()
      absl.flags.FLAGS(sys.argv, known_only=True)
      self.complete_absl_config(absl.flags)
      already_configured_with_absl = True


class NameSpace(object):
  # [CLASS DOC]
  # Tiny attribute-access facade: any attribute read is forwarded to the getter callback given at
  # construction, so config option values can be read like attributes. Used as the FLAGS object.
  def __init__(self, getter):
    # [FUNCTION DOC]
    # Stores the getter callback (taking an option name and returning its value). Returns nothing.
    self._getter = getter

  def __getattr__(self, name):
    # [FUNCTION DOC]
    # Called for attributes not found normally; returns the result of the getter callback for that
    # name (which may raise if the option is unknown).
    return self._getter(name)


config = Config()
flags = config
FLAGS = flags.FLAGS

already_configured_with_absl = False

flags.DEFINE_bool(
    'jax_enable_checks',
    bool_env('JAX_ENABLE_CHECKS', False),
    help='Turn on invariant checking (core.skip_checks = False)')

flags.DEFINE_bool('tf_numpy_additional_tests', True,
                  'Run tests added specifically for TF numpy')
