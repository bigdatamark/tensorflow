# [MODULE DOC]
# Building blocks for the global AutoGraph configuration: an action enumeration (none, convert,
# do not convert) and a small hierarchy of rules that decide, by module name prefix, which action
# applies to a module. The concrete rule list lives in the sibling config module. No import-time
# side effects.
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
"""Global configuration support."""

import enum


# TODO(mdan): For better performance, allow each rule to take a set names.


class Rule(object):
  # [CLASS DOC]
  # Base class for conversion rules. Holds a module name prefix and offers a name-matching test.
  # Subclasses add the action to take when the rule matches.
  """Base class for conversion rules."""

  def __init__(self, module_prefix):
    # [FUNCTION DOC]
    # Constructor. Input: module_prefix, a dotted module name string. Stores it privately.
    self._prefix = module_prefix

  def matches(self, module_name):
    # [FUNCTION DOC]
    # Input: a dotted module name string. Returns true if the name equals the rule prefix or is a
    # submodule of it (starts with the prefix followed by a dot); a mere string prefix such as a
    # longer sibling name does not match.
    return (module_name.startswith(self._prefix + '.') or
            module_name == self._prefix)


class Action(enum.Enum):
  # [CLASS DOC]
  # Enumeration of conversion decisions a rule can yield: none (rule does not apply, zero),
  # convert (one) and do not convert (two).
  NONE = 0
  CONVERT = 1
  DO_NOT_CONVERT = 2


class DoNotConvert(Rule):
  # [CLASS DOC]
  # Rule saying modules under its prefix must not be converted by AutoGraph. Created with a
  # module name prefix; used in the global rule list.
  """Indicates that this module should be not converted."""

  def __str__(self):
    # [FUNCTION DOC]
    # Returns a human-readable description naming this rule type and its prefix.
    return 'DoNotConvert rule for {}'.format(self._prefix)

  def get_action(self, module):
    # [FUNCTION DOC]
    # Input: a module object. Returns the do-not-convert action if the module's name matches the
    # rule prefix, otherwise the none action.
    if self.matches(module.__name__):
      return Action.DO_NOT_CONVERT
    return Action.NONE


class Convert(Rule):
  # [CLASS DOC]
  # Rule saying modules under its prefix must be converted by AutoGraph. Created with a module
  # name prefix; used in the global rule list.
  """Indicates that this module should be converted."""

  def __str__(self):
    # [FUNCTION DOC]
    # Returns a human-readable description naming this rule type and its prefix.
    return 'Convert rule for {}'.format(self._prefix)

  def get_action(self, module):
    # [FUNCTION DOC]
    # Input: a module object. Returns the convert action if the module's name matches the rule
    # prefix, otherwise the none action.
    if self.matches(module.__name__):
      return Action.CONVERT
    return Action.NONE
