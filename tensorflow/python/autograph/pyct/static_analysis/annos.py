# [MODULE DOC]
# Defines the NodeAnno enumeration of extra annotation keys used by the static analyzers
# (symbol flags and scope kinds such as arguments, condition, iterate, body, orelse and
# arguments-and-body scopes), complementing the basic keys in the anno module. Helper module
# for activity and other static analysis modules. No import-time side effects beyond enum creation.
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
"""Annotations used by the static analyzer."""

from enum import Enum


# TODO(mdan): Remove.


class NoValue(Enum):
  # [CLASS DOC]
  # HELPER: base enumeration whose members print as their bare names; used by NodeAnno.

  def __repr__(self):  # pylint: disable=invalid-repr-returned
    # [FUNCTION DOC]
    # Returns the member name as its text representation.
    return self.name


class NodeAnno(NoValue):
  # [CLASS DOC]
  # Enumeration of annotation keys: symbol flags (local, parameter, modified since entry) and
  # scope kinds attached to nodes (arguments, condition, iterate, arguments-and-body, body,
  # orelse). Each member's value is a human-readable description. Used with the anno module.
  """Additional annotations used by the static analyzer.

  These are in addition to the basic annotations declared in anno.py.
  """

  # Symbols
  # These flags are boolean.
  IS_LOCAL = 'Symbol is local to the function scope being analyzed.'
  IS_PARAM = 'Symbol is a parameter to the function being analyzed.'
  IS_MODIFIED_SINCE_ENTRY = (
      'Symbol has been explicitly replaced in the current function scope.')

  # Scopes
  # Scopes are represented by objects of type activity.Scope.
  ARGS_SCOPE = 'The scope for the argument list of a function call.'
  COND_SCOPE = 'The scope for the test node of a conditional statement.'
  ITERATE_SCOPE = 'The scope for the iterate assignment of a for loop.'
  ARGS_AND_BODY_SCOPE = (
      'The scope for the main body of a function or lambda, including its'
      ' arguments.')
  BODY_SCOPE = (
      'The scope for the main body of a statement (True branch for if '
      'statements, main body for loops).')
  ORELSE_SCOPE = (
      'The scope for the orelse body of a statement (False branch for if '
      'statements, orelse body for loops).')
