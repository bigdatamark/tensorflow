# [MODULE DOC]
# AST node annotation support (adapted from Tangent). Annotations are stored in a dictionary kept
# in a special attribute of a syntax tree node; the module offers functions to list, get, test,
# set, delete, copy and duplicate annotations, plus enumerations naming the standard annotation
# keys (basic keys and static-analysis keys). The attribute is also registered in the node's field
# list so annotations survive conversions between the gast and ast representations.
# Used throughout the pyct analyses and transformers. Defines a module-level sentinel for
# "no default given".
#
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
"""AST node annotation support.

Adapted from Tangent.
"""

import enum

# pylint:disable=g-bad-import-order

import gast
# pylint:enable=g-bad-import-order


# TODO(mdan): Shorten the names.
# These names are heavily used, and anno.blaa
# TODO(mdan): Replace the attr-dict mechanism with a more typed solution.


class NoValue(enum.Enum):
  # [CLASS DOC]
  # Base enumeration for annotation key types. Members' values are documentation strings only.
  # Adds convenience methods so a key can read, write and test its own annotation on a node, and a
  # representation that shows just the member name.
  """Base class for different types of AST annotations."""

  def of(self, node, default=None):
    # [FUNCTION DOC]
    # Returns the annotation of this key on node, or default (null by default) when missing.
    # Unlike the module-level getter, the default is never the "must exist" sentinel, so it does
    # not raise when the annotation is absent.
    return getanno(node, self, default=default)

  def add_to(self, node, value):
    # [FUNCTION DOC]
    # Sets the annotation of this key on node to value (mutates node). Returns nothing.
    setanno(node, self, value)

  def exists(self, node):
    # [FUNCTION DOC]
    # Returns true if node carries an annotation for this key.
    return hasanno(node, self)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the member name as text (without the enumeration class name or value).
    return str(self.name)


class Basic(NoValue):
  # [CLASS DOC]
  # Enumeration of basic annotation keys: qualified name, skip-processing marker, indent-block
  # remainder, origin information, user directives and extra loop test. Values are descriptions.
  """Container for basic annotation keys.

  The enum values are used strictly for documentation purposes.
  """

  QN = 'Qualified name, as it appeared in the code. See qual_names.py.'
  SKIP_PROCESSING = (
      'This node should be preserved as is and not processed any further.')
  INDENT_BLOCK_REMAINDER = (
      'When a node is annotated with this, the remainder of the block should'
      ' be indented below it. The annotation contains a tuple'
      ' (new_body, name_map), where `new_body` is the new indented block and'
      ' `name_map` allows renaming symbols.')
  ORIGIN = ('Information about the source code that converted code originated'
            ' from. See origin_information.py.')
  DIRECTIVES = ('User directives associated with a statement or a variable.'
                ' Typically, they affect the immediately-enclosing statement.')

  EXTRA_LOOP_TEST = (
      'A special annotation containing additional test code to be executed in'
      ' for loops.')


class Static(NoValue):
  # [CLASS DOC]
  # Enumeration of annotation keys produced by static analyses: parameter flag, scopes (general,
  # arguments, condition, body, orelse), reaching definitions, defined functions and variables,
  # live variables in/out, static types, closure types and static values. Values are descriptions.
  """Container for static analysis annotation keys.

  The enum values are used strictly for documentation purposes.
  """

  # Symbols
  # These flags are boolean.
  IS_PARAM = 'Symbol is a parameter to the function being analyzed.'

  # Scopes
  # Scopes are represented by objects of type activity.Scope.
  SCOPE = 'The scope for the annotated node. See activity.py.'
  # TODO(mdan): Drop these in favor of accessing the child's SCOPE.
  ARGS_SCOPE = 'The scope for the argument list of a function call.'
  COND_SCOPE = 'The scope for the test node of a conditional statement.'
  BODY_SCOPE = (
      'The scope for the main body of a statement (True branch for if '
      'statements, main body for loops).')
  ORELSE_SCOPE = (
      'The scope for the orelse body of a statement (False branch for if '
      'statements, orelse body for loops).')

  # Static analysis annotations.
  DEFINITIONS = (
      'Reaching definition information. See reaching_definitions.py.')
  ORIG_DEFINITIONS = (
      'The value of DEFINITIONS that applied to the original code before any'
      ' conversion.')
  DEFINED_FNS_IN = (
      'Local function definitions that may exist when exiting the node. See'
      ' reaching_fndefs.py')
  DEFINED_VARS_IN = (
      'Symbols defined when entering the node. See reaching_definitions.py.')
  LIVE_VARS_OUT = ('Symbols live when exiting the node. See liveness.py.')
  LIVE_VARS_IN = ('Symbols live when entering the node. See liveness.py.')
  TYPES = 'Static type information. See type_inference.py.'
  CLOSURE_TYPES = 'Types of closure symbols at each detected call site.'
  VALUE = 'Static value information. See type_inference.py.'


FAIL = object()


def keys(node, field_name='___pyct_anno'):
  # [FUNCTION DOC]
  # Returns an immutable set of the annotation keys present on node, stored under the attribute
  # named field_name (default is the standard annotation attribute). Empty if node has none.
  if not hasattr(node, field_name):
    return frozenset()
  return frozenset(getattr(node, field_name).keys())


def getanno(node, key, default=FAIL, field_name='___pyct_anno'):
  # [FUNCTION DOC]
  # Returns the annotation for key on node. If default is not given (sentinel) a missing
  # annotation raises an attribute error (no annotation attribute) or key error (missing key);
  # if a default was given it is returned when the annotation is missing. field_name selects the
  # annotation attribute.
  if (default is FAIL or (hasattr(node, field_name) and
                          (key in getattr(node, field_name)))):
    return getattr(node, field_name)[key]
  return default


def hasanno(node, key, field_name='___pyct_anno'):
  # [FUNCTION DOC]
  # Returns true if node has the annotation attribute (field_name) and it contains key.
  return hasattr(node, field_name) and key in getattr(node, field_name)


def setanno(node, key, value, field_name='___pyct_anno'):
  # [FUNCTION DOC]
  # Stores value under key in node's annotation mapping (created if absent), overwriting any
  # previous value. Side effect: mutates node, and appends field_name to the node's field list when
  # not already present so annotations survive tree conversions. Returns nothing.
  annotations = getattr(node, field_name, {})
  setattr(node, field_name, annotations)
  annotations[key] = value

  # So that the annotations survive gast_to_ast() and ast_to_gast()
  if field_name not in node._fields:
    node._fields += (field_name,)


def delanno(node, key, field_name='___pyct_anno'):
  # [FUNCTION DOC]
  # Removes the annotation key from node. Raises an attribute error if node has no annotations and
  # a key error if the key is absent. When the last annotation is removed, also deletes the
  # annotation attribute and removes it from the node's field list. Mutates node.
  annotations = getattr(node, field_name)
  del annotations[key]
  if not annotations:
    delattr(node, field_name)
    node._fields = tuple(f for f in node._fields if f != field_name)


def copyanno(from_node, to_node, key, field_name='___pyct_anno'):
  # [FUNCTION DOC]
  # Copies the annotation key from from_node to to_node (same field_name) if it exists on the
  # source; does nothing otherwise. The value is shared, not deep-copied. Mutates to_node.
  if hasanno(from_node, key, field_name=field_name):
    setanno(
        to_node,
        key,
        getanno(from_node, key, field_name=field_name),
        field_name=field_name)


def dup(node, copy_map, field_name='___pyct_anno'):
  # [FUNCTION DOC]
  # Walks node and all its descendants; on each, for every source key in copy_map that is
  # annotated, sets an annotation under the mapped destination key with the same value (the source
  # annotation is kept). Mutates the tree. Returns nothing.
  """Recursively copies annotations in an AST tree.

  Args:
    node: ast.AST
    copy_map: Dict[Hashable, Hashable], maps a source anno key to a destination
        key. All annotations with the source key will be copied to identical
        annotations with the destination key.
    field_name: str
  """
  for n in gast.walk(node):
    for k in copy_map:
      if hasanno(n, k, field_name):
        setanno(n, copy_map[k], getanno(n, k, field_name), field_name)
