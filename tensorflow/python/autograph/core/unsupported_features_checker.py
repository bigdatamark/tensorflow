# [MODULE DOC]
# Early checker for Python features AutoGraph knows it cannot convert: name-mangled attributes,
# for/else and while/else statements, and generators (yield and yield from). Exposes a node
# visitor class and a verify function that raises an unsupported-language-element error when any
# is found, which prevents the function from being compiled. No import-time side effects.
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
"""Checkers for detecting unsupported Python features."""

import gast

from tensorflow.python.autograph.pyct import errors


class UnsupportedFeaturesChecker(gast.NodeVisitor):
  # [CLASS DOC]
  # AST node visitor that walks a function's tree and raises an unsupported-language-element
  # error at the first unsupported construct. Stateless; created and used by the verify function.
  """Quick check for Python features we know we don't support.

  Any features detected will cause AutoGraph to not compile a function.
  """

  def visit_Attribute(self, node):
    # [FUNCTION DOC]
    # Visitor for attribute accesses. If the attribute name starts with two underscores but does
    # not end with two (a name-mangled private name), raises an unsupported-language-element
    # error saying mangled names are not supported. Otherwise continues into the children.
    if (node.attr is not None
        and node.attr.startswith('__') and not node.attr.endswith('__')):
      raise errors.UnsupportedLanguageElementError(
          'mangled names are not yet supported')
    self.generic_visit(node)

  def visit_For(self, node):
    # [FUNCTION DOC]
    # Visitor for for loops. Raises an unsupported-language-element error if the loop has an else
    # clause; otherwise continues into the children.
    if node.orelse:
      raise errors.UnsupportedLanguageElementError(
          'for/else statement not yet supported')
    self.generic_visit(node)

  def visit_While(self, node):
    # [FUNCTION DOC]
    # Visitor for while loops. Raises an unsupported-language-element error if the loop has an
    # else clause; otherwise continues into the children.
    if node.orelse:
      raise errors.UnsupportedLanguageElementError(
          'while/else statement not yet supported')
    self.generic_visit(node)

  # These checks could potentially be replaced with inspect.isgeneratorfunction
  # to avoid a getsource/parse/ast-walk round trip.
  def visit_Yield(self, node):
    # [FUNCTION DOC]
    # Visitor for yield expressions: always raises an unsupported-language-element error saying
    # generators are not supported.
    raise errors.UnsupportedLanguageElementError('generators are not supported')

  def visit_YieldFrom(self, node):
    # [FUNCTION DOC]
    # Visitor for yield-from expressions: always raises an unsupported-language-element error
    # saying generators are not supported.
    raise errors.UnsupportedLanguageElementError('generators are not supported')


def verify(node):
  # [FUNCTION DOC]
  # Entry point. Input: the AST node of a function. Runs the unsupported features checker over it;
  # raises an unsupported-language-element error if anything unsupported is found. Returns nothing.
  UnsupportedFeaturesChecker().visit(node)
