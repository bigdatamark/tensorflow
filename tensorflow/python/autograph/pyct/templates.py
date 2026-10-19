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
# [MODULE DOC]
# Template-based AST generation for autograph: parse a snippet of source text containing
# placeholder symbols and substitute them with AST nodes, names or qualified names.
# Exposes: ContextAdjuster (fixes read/write context of substituted nodes), ReplaceTransformer
# (does the substitution), and the functions replace and replace_as_expression. Underscore
# function _convert_to_ast normalizes replacement values. Used by many autograph converters.
# No import-time side effects.
"""AST conversion templates.

Adapted from Tangent.
"""

import ast
import textwrap

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names


class ContextAdjuster(gast.NodeTransformer):
  # [CLASS DOC]
  # Transformer that rewrites the read/write context field of names, tuples, lists, attributes
  # and subscripts to a requested context class, so substituted nodes match their position in a
  # template. Holds one piece of state, the current override (a context class or nothing). The
  # override is reset to nothing below calls, dicts, comprehensions and lambdas (those contexts
  # are only checked, not changed) and set to load below attribute and subscript bases.
  """Adjusts the ctx field of nodes to ensure consistency.

  This transformer can change the ctx fields of a variable, tuple and other
  AST elements that allow one, based on whether the element is being read or
  written.
  """

  def __init__(self, override_value):
    # [FUNCTION DOC]
    # Stores override_value, a context class (load, store, delete) used to build new context
    # objects, or nothing to disable overriding. Returns nothing.
    self._ctx_override = override_value

  def visit(self, node):
    # [FUNCTION DOC]
    # Visits a node with the normal dispatch, then asserts that any node having a context field
    # has it set (assertion error otherwise), and restores the override that was active on entry
    # so changes made while visiting children do not leak to siblings. Returns the visited node.
    original_override = self._ctx_override
    node = super(ContextAdjuster, self).visit(node)
    if hasattr(node, 'ctx'):
      assert node.ctx is not None, 'node {} has ctx unset'.format(node)
    self._ctx_override = original_override
    return node

  def _apply_override(self, node):
    # [FUNCTION DOC]
    # HELPER: shared by the visit methods of ContextAdjuster.
    # If an override is active, replaces the node's context with a fresh instance of it (mutates
    # the node). Returns nothing.
    if self._ctx_override is not None:
      node.ctx = self._ctx_override()

  def visit_Attribute(self, node):
    # [FUNCTION DOC]
    # Applies the override to the attribute node, then sets the override to load for its
    # children (the object being accessed is only read) and visits them. Returns the node.
    self._apply_override(node)
    self._ctx_override = gast.Load
    node = self.generic_visit(node)
    return node

  def visit_Tuple(self, node):
    # [FUNCTION DOC]
    # Applies the override to the tuple and keeps it active while visiting the elements.
    # Returns the node.
    self._apply_override(node)
    return self.generic_visit(node)

  def visit_List(self, node):
    # [FUNCTION DOC]
    # Applies the override to the list and keeps it active while visiting the elements.
    # Returns the node.
    self._apply_override(node)
    return self.generic_visit(node)

  def visit_Name(self, node):
    # [FUNCTION DOC]
    # Applies the override to the name node, then visits children. Returns the node.
    self._apply_override(node)
    return self.generic_visit(node)

  def visit_Call(self, node):
    # [FUNCTION DOC]
    # Applies the override to the call node itself, then disables overriding for its children
    # (they must already have contexts, which the visit method asserts). Returns the node.
    self._apply_override(node)
    # We may be able to override these to Load(), but for now it's simpler
    # to just assert that they're set.
    self._ctx_override = None
    return self.generic_visit(node)

  def visit_Dict(self, node):
    # [FUNCTION DOC]
    # Disables overriding for the dict's children (their contexts are only asserted to be set)
    # and visits them. Returns the node.
    # We may be able to override these to Load(), but for now it's simpler
    # to just assert that they're set.
    self._ctx_override = None
    return self.generic_visit(node)

  def visit_Subscript(self, node):
    # [FUNCTION DOC]
    # Applies the override to the subscript node, sets the override to load, visits the indexed
    # value explicitly, then visits all children generically (so the value is visited twice).
    # Returns the node.
    self._apply_override(node)
    self._ctx_override = gast.Load
    node.value = self.visit(node.value)
    return self.generic_visit(node)

  def visit_comprehension(self, node):
    # [FUNCTION DOC]
    # Disables overriding below a comprehension clause and visits its children. Returns the node.
    # We may be able to override some of these, but for now it's simpler
    # to just assert that they're set.
    self._ctx_override = None
    return self.generic_visit(node)

  def visit_Lambda(self, node):
    # [FUNCTION DOC]
    # Disables overriding below a lambda and visits its children. Returns the node.
    # We may be able to override some of these, but for now it's simpler
    # to just assert that they're set.
    self._ctx_override = None
    return self.generic_visit(node)


class ReplaceTransformer(gast.NodeTransformer):
  # [CLASS DOC]
  # Transformer that substitutes placeholder names, keyword arguments, function names and
  # attribute names in a parsed template with caller-supplied replacement nodes. State: the
  # replacements mapping, an unused in-replacements flag, and the set of annotation keys kept
  # when replacement nodes are copied. Used by the replace function, one instance per node.
  """Replace AST nodes."""

  def __init__(self, replacements):
    # [FUNCTION DOC]
    # Stores the replacements mapping (placeholder name to node or list of nodes), initializes an
    # unused flag, and defines the set of annotation keys preserved when copying replacement
    # nodes (directives, extra loop test, origin, skip-processing, original definitions and the
    # function context name). Returns nothing.
    """Create a new ReplaceTransformer.

    Args:
      replacements: A mapping from placeholder names to (lists of) AST nodes
          that these placeholders will be replaced by.
    """
    self.replacements = replacements
    self.in_replacements = False
    self.preserved_annos = {
        anno.Basic.DIRECTIVES,
        anno.Basic.EXTRA_LOOP_TEST,
        anno.Basic.ORIGIN,
        anno.Basic.SKIP_PROCESSING,
        anno.Static.ORIG_DEFINITIONS,
        'function_context_name',
    }

  def _prepare_replacement(self, replaced, key):
    # [FUNCTION DOC]
    # HELPER: fetches and cleans replacement nodes for the visit methods of ReplaceTransformer.
    # Looks up the replacement for key, makes a clean copy (keeping only the preserved
    # annotations) and returns it as a list (a single node is wrapped in a list). The replaced
    # argument is unused. Raises a missing-key error if key is absent.
    """Prepares a replacement AST that's safe to swap in for a node.

    Args:
      replaced: ast.AST, the node being replaced
      key: Hashable, the key of the replacement AST
    Returns:
      ast.AST, the replacement AST
    """
    repl = self.replacements[key]

    new_nodes = ast_util.copy_clean(repl, preserve_annos=self.preserved_annos)
    if isinstance(new_nodes, gast.AST):
      new_nodes = [new_nodes]

    return new_nodes

  def visit_Expr(self, node):
    # [FUNCTION DOC]
    # Visits the wrapped expression. If it was replaced, returns the replacement directly (so a
    # statement placeholder is not wrapped in an expression statement); otherwise returns the
    # original node.
    # When replacing a placeholder with an entire statement, the replacement
    # must stand on its own and not be wrapped in an Expr.
    new_value = self.visit(node.value)
    if new_value is node.value:
      return node
    return new_value

  def visit_keyword(self, node):
    # [FUNCTION DOC]
    # Replaces a keyword argument whose name is a placeholder. The replacement must be a keyword
    # node or a non-empty list/tuple of keyword nodes, which is returned. Non-placeholder
    # keywords are visited normally. Raises ValueError for any other replacement kind.
    if node.arg not in self.replacements:
      return self.generic_visit(node)

    repl = self._prepare_replacement(node, node.arg)
    if isinstance(repl, gast.keyword):
      return repl
    elif (repl and isinstance(repl, (list, tuple)) and
          all(isinstance(r, gast.keyword) for r in repl)):
      return repl
    # TODO(mdan): We may allow replacing with a string as well.
    # For example, if one wanted to replace foo with bar in foo=baz, then
    # we could allow changing just node arg, so that we end up with bar=baz.
    raise ValueError(
        'a keyword argument may only be replaced by another keyword or a '
        'non-empty list of keywords. Found: {} for keyword {}'.format(
            repl, node.arg))

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Visits children, then if the function's name is a placeholder, renames the function to the
    # identifier of the replacement, which must be a name node. Raises ValueError otherwise.
    # Returns the node.
    node = self.generic_visit(node)
    if node.name not in self.replacements:
      return node

    repl = self.replacements[node.name]
    if not isinstance(repl, (gast.Name, ast.Name)):
      raise ValueError(
          'a function name can only be replaced by a Name node. Found: %s' %
          repl)
    node.name = repl.id
    return node

  def visit_Attribute(self, node):
    # [FUNCTION DOC]
    # Visits children, then if the attribute name is a placeholder, replaces it with the
    # identifier of the replacement, which must be a gast name node (ValueError otherwise).
    # Returns the node.
    node = self.generic_visit(node)
    if node.attr not in self.replacements:
      return node

    repl = self.replacements[node.attr]
    if not isinstance(repl, gast.Name):
      raise ValueError(
          'An attribute can only be replaced by a Name node. Found: %s' % repl)
    node.attr = repl.id
    return node

  def visit_Name(self, node):
    # [FUNCTION DOC]
    # If the name is a placeholder, returns its replacement: a cleaned copy of the replacement
    # nodes, with their read/write context adjusted to that of the original name. An empty
    # replacement list yields the empty value (the node is removed). A single replacement is
    # returned as one node, several as a list. Non-placeholder names are returned unchanged.
    if node.id not in self.replacements:
      return node

    new_nodes = self._prepare_replacement(node, node.id)

    if not new_nodes:
      return new_nodes

    # Preserve the target context.
    adjuster = ContextAdjuster(type(node.ctx))
    for n in new_nodes:
      if hasattr(n, 'ctx'):
        adjuster.visit(n)

    if len(new_nodes) == 1:
      new_nodes, = new_nodes

    return new_nodes


def _convert_to_ast(n):
  # [FUNCTION DOC]
  # HELPER: normalizes replacement values for the replace function.
  # Converts a string into a name node with unset context, a QN into its AST form, lists and
  # tuples element-wise (recursively, preserving the container kind), and returns anything else
  # unchanged. Pure.
  """Converts from a known data type to AST."""
  # Note: When generating AST nodes from strings/QNs in isolation, ctx is
  # unknown. ctx must be filled in according to the template being used.
  # See ReplaceTransformer.visit_Name.
  if isinstance(n, str):
    return gast.Name(id=n, ctx=None, annotation=None, type_comment=None)
  if isinstance(n, qual_names.QN):
    return n.ast()
  if isinstance(n, list):
    return [_convert_to_ast(e) for e in n]
  if isinstance(n, tuple):
    return tuple(_convert_to_ast(e) for e in n)
  return n


def replace(template, **replacements):
  # [FUNCTION DOC]
  # Main entry point. template is a string of source code (indented text is dedented);
  # replacements maps placeholder names to nodes, lists of nodes, strings or QNs. Converts the
  # replacements to AST, prepends the standard preamble, parses the template into statements,
  # applies a ReplaceTransformer to each, flattens list results, and resolves qualified names on
  # every result. Returns a list of nodes. Mutates the passed replacements mapping values to their
  # AST forms. Raises ValueError if the template is not a string.
  # KNOWN ISSUE: the docstring claims a node or single AST is returned in some cases, but the
  # code always returns a list of nodes.
  """Replaces placeholders in a Python template.

  AST Name and Tuple nodes always receive the context that inferred from
  the template. However, when replacing more complex nodes (that can potentially
  contain Name children), then the caller is responsible for setting the
  appropriate context.

  Args:
    template: A string representing Python code. Any symbol name can be used
        that appears in the template code can be used as placeholder.
    **replacements: A mapping from placeholder names to (lists of) AST nodes
        that these placeholders will be replaced by. String values are also
        supported as a shorthand for AST Name nodes with the respective ID.

  Returns:
    An AST node or list of AST nodes with the replacements made. If the
    template was a function, a list will be returned. If the template was a
    node, the same node will be returned. If the template was a string, an
    AST node will be returned (a `Module` node in the case of a multi-line
    string, an `Expr` node otherwise).

  Raises:
    ValueError: if the arguments are incorrect.
  """
  if not isinstance(template, str):
    raise ValueError('Expected string template, got %s' % type(template))
  for k in replacements:
    replacements[k] = _convert_to_ast(replacements[k])
  template_str = parser.STANDARD_PREAMBLE + textwrap.dedent(template)
  nodes = parser.parse(
      template_str,
      preamble_len=parser.STANDARD_PREAMBLE_LEN,
      single_node=False)
  results = []
  for node in nodes:
    node = ReplaceTransformer(replacements).visit(node)
    if isinstance(node, (list, tuple)):
      results.extend(node)
    else:
      results.append(node)
  results = [qual_names.resolve(r) for r in results]
  return results


def replace_as_expression(template, **replacements):
  # [FUNCTION DOC]
  # Like replace, but the template must produce exactly one expression. Returns the expression
  # node (the value of an expression statement, or a bare name node). Raises ValueError if the
  # result is not a single node, or if that node is neither an expression statement nor a name.
  """Variant of replace that generates expressions, instead of code blocks."""
  replacement = replace(template, **replacements)
  if len(replacement) != 1:
    raise ValueError(
        'single expression expected; for more general templates use replace')
  node, = replacement

  if isinstance(node, gast.Expr):
    return node.value
  elif isinstance(node, gast.Name):
    return node

  raise ValueError(
      'the template is expected to generate an expression or a name node;'
      ' instead found %s' % node)
