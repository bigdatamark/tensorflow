# [MODULE DOC]
# Utilities for manipulating syntax trees (gast/ast): a clean deep copier (copy_clean), a symbol
# renamer driven by qualified-name annotations (rename_symbols), conversion of keyword nodes to a
# dictionary node, a wildcard-capable structural pattern matcher (matches), a helper that breaks
# possibly-unpacked assignments into single assignments (apply_to_single_assignments) and a
# generator that walks two identically structured trees in parallel (parallel_walk).
# Depends on the anno, parser and qual_names modules of the same package. No import-time effects.
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
"""AST manipulation utilities."""

import ast

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names


class CleanCopier(object):
  # [CLASS DOC]
  # HELPER: implementation behind copy_clean.
  # Visitor-like deep copier for syntax trees. State: preserve_annos, an optional collection of
  # annotation keys whose annotations are carried over to the copy. Fields whose names start with
  # two underscores (including the annotation field) are not copied, other than the preserved keys.
  """NodeTransformer-like visitor that copies an AST."""

  def __init__(self, preserve_annos):
    # [FUNCTION DOC]
    # Constructor. preserve_annos is a collection of annotation keys to keep on copies, or null
    # for none. Stores it on the instance.
    super(CleanCopier, self).__init__()
    self.preserve_annos = preserve_annos

  def copy(self, node):
    # [FUNCTION DOC]
    # Recursively copies node. Lists and tuples are copied element-wise into the same container
    # kind; values that are not syntax nodes are returned as is (shared); syntax nodes are rebuilt
    # from the copies of their present, non-double-underscore fields, then the preserved
    # annotation keys are copied from the original to the new node. Returns the copy.
    """Returns a deep copy of node (excluding some fields, see copy_clean)."""

    if isinstance(node, list):
      return [self.copy(n) for n in node]
    elif isinstance(node, tuple):
      return tuple(self.copy(n) for n in node)
    elif not isinstance(node, (gast.AST, ast.AST)):
      # Assuming everything that's not an AST, list or tuple is a value type
      # and may simply be assigned.
      return node

    assert isinstance(node, (gast.AST, ast.AST))

    new_fields = {}
    for f in node._fields:
      if not f.startswith('__') and hasattr(node, f):
        new_fields[f] = self.copy(getattr(node, f))
    new_node = type(node)(**new_fields)

    if self.preserve_annos:
      for k in self.preserve_annos:
        anno.copyanno(node, new_node, k)
    return new_node


def copy_clean(node, preserve_annos=None):
  # [FUNCTION DOC]
  # Returns a deep copy of the syntax tree node without double-underscore fields, keeping only
  # annotations whose keys are listed in preserve_annos (optional, default none). Does not mutate
  # the input.
  """Creates a deep copy of an AST.

  The copy will not include fields that are prefixed by '__', with the
  exception of user-specified annotations.

  Args:
    node: ast.AST
    preserve_annos: Optional[Set[Hashable]], annotation keys to include in the
        copy
  Returns:
    ast.AST
  """
  return CleanCopier(preserve_annos).copy(node)


class SymbolRenamer(gast.NodeTransformer):
  # [CLASS DOC]
  # HELPER: implementation behind rename_symbols.
  # Tree transformer that renames symbols according to name_map, a mapping from qualified names to
  # replacement names. Handles name nodes, attribute chains carrying a qualified-name annotation,
  # nonlocal/global name lists and function definition names. Renaming of plain attributes without
  # a qualified-name annotation is not supported. Requires qualified-name annotations on the tree.
  """Transformer that can rename symbols to a simple names."""

  def __init__(self, name_map):
    # [FUNCTION DOC]
    # Constructor. name_map is a mapping from qualified names to the new names (converted to text
    # when applied). Stores it on the instance.
    self.name_map = name_map

  def _process_name_node(self, node):
    # [FUNCTION DOC]
    # HELPER: shared by the name and attribute visitors.
    # Reads the qualified-name annotation of node (required; raises if absent). If it is in the
    # name map, returns a new simple name node with the mapped name, the same context, and all
    # annotations of the original copied over. Otherwise visits the children and returns node.
    qn = anno.getanno(node, anno.Basic.QN)
    if qn in self.name_map:
      new_node = gast.Name(
          str(self.name_map[qn]),
          ctx=node.ctx,
          annotation=None,
          type_comment=None)
      # All annotations get carried over.
      for k in anno.keys(node):
        anno.copyanno(node, new_node, k)
      return new_node
    return self.generic_visit(node)

  def _process_list_of_strings(self, names):
    # [FUNCTION DOC]
    # HELPER: renaming for nonlocal/global statements.
    # For each text name in the list names, if its qualified name is in the name map, replaces the
    # entry in place with the mapped name. Returns the same (mutated) list.
    for i in range(len(names)):
      qn = qual_names.QN(names[i])
      if qn in self.name_map:
        names[i] = str(self.name_map[qn])
    return names

  def visit_Nonlocal(self, node):
    # [FUNCTION DOC]
    # Renames the names listed in a nonlocal statement per the name map (mutates and returns node).
    node.names = self._process_list_of_strings(node.names)
    return node

  def visit_Global(self, node):
    # [FUNCTION DOC]
    # Renames the names listed in a global statement per the name map (mutates and returns node).
    node.names = self._process_list_of_strings(node.names)
    return node

  def visit_Name(self, node):
    # [FUNCTION DOC]
    # Renames a name node via the shared name processing; returns the possibly new node.
    return self._process_name_node(node)

  def visit_Attribute(self, node):
    # [FUNCTION DOC]
    # If the attribute node has a qualified-name annotation, processes it like a name (the whole
    # chain may be replaced by a simple name); otherwise just visits its children without renaming.
    if anno.hasanno(node, anno.Basic.QN):
      return self._process_name_node(node)
    # Renaming attributes is not supported.
    return self.generic_visit(node)

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Renames the function's own name if the name map has an entry for it (mutates node), then
    # visits the children (so body and decorators are renamed too). Returns the node.
    qn = qual_names.QN(node.name)
    if qn in self.name_map:
      node.name = str(self.name_map[qn])
    return self.generic_visit(node)


def rename_symbols(node, name_map):
  # [FUNCTION DOC]
  # Renames symbols in a syntax tree. node may be a single node, a list or a tuple of nodes (the
  # returned container keeps the kind); name_map maps qualified names to new names. The tree is
  # mutated in place where possible. Requires qualified-name annotations. Returns the renamed
  # node(s).
  """Renames symbols in an AST. Requires qual_names annotations."""
  renamer = SymbolRenamer(name_map)
  if isinstance(node, list):
    return [renamer.visit(n) for n in node]
  elif isinstance(node, tuple):
    return tuple(renamer.visit(n) for n in node)
  return renamer.visit(node)


def keywords_to_dict(keywords):
  # [FUNCTION DOC]
  # Converts a list of keyword-argument nodes into a dictionary-literal node whose keys are
  # constant nodes holding the argument names and whose values are the original value nodes.
  # Returns the dictionary node.
  """Converts a list of ast.keyword objects to a dict."""
  keys = []
  values = []
  for kw in keywords:
    keys.append(gast.Constant(kw.arg, kind=None))
    values.append(kw.value)
  return gast.Dict(keys=keys, values=values)


class PatternMatcher(gast.NodeVisitor):
  # [CLASS DOC]
  # HELPER: implementation behind matches.
  # Visitor that walks a node alongside a pattern tree. State: pattern (the current pattern node),
  # pattern_stack (saved patterns while descending) and matches (flag, true until a mismatch is
  # found). A name node with identifier underscore, or a single-item list of one, in the pattern
  # is a wildcard matching anything.
  """Matches a node against a pattern represented by a node."""

  def __init__(self, pattern):
    # [FUNCTION DOC]
    # Constructor. pattern is the pattern syntax tree. Initializes an empty pattern stack and sets
    # the match flag to true.
    self.pattern = pattern
    self.pattern_stack = []
    self.matches = True

  def compare_and_visit(self, node, pattern):
    # [FUNCTION DOC]
    # HELPER: recursion step of the matcher.
    # Temporarily makes pattern the current pattern (saving the previous one on the stack), visits
    # the children of node, then restores the previous pattern. Returns nothing.
    self.pattern_stack.append(self.pattern)
    self.pattern = pattern
    self.generic_visit(node)
    self.pattern = self.pattern_stack.pop()

  def no_match(self):
    # [FUNCTION DOC]
    # HELPER: records a mismatch. Sets the match flag to false and returns false.
    self.matches = False
    return False

  def is_wildcard(self, p):
    # [FUNCTION DOC]
    # HELPER: wildcard test. A one-element list or tuple is unwrapped first. Returns true if the
    # result is a name node with identifier underscore or the text underscore; false otherwise.
    if isinstance(p, (list, tuple)) and len(p) == 1:
      p, = p
    if isinstance(p, gast.Name) and p.id == '_':
      return True
    if p == '_':
      return True
    return False

  def generic_visit(self, node):
    # [FUNCTION DOC]
    # Compares node against the current pattern field by field (skipping double-underscore
    # fields). Does nothing if a mismatch was already recorded. A mismatch is recorded when: a
    # field is missing on the node but non-empty in the pattern, or missing in the pattern;
    # list lengths differ or kinds differ; syntax node types are unrelated by inheritance; or value
    # fields are unequal. Wildcard pattern fields are skipped. Matching list items and child nodes
    # are compared recursively. Returns false on some mismatches, nothing otherwise; the result
    # is read from the match flag.
    if not self.matches:
      return

    pattern = self.pattern
    for f in node._fields:
      if f.startswith('__'):
        continue

      if not hasattr(node, f):
        if hasattr(pattern, f) and getattr(pattern, f):
          return self.no_match()
        else:
          continue
      if not hasattr(pattern, f):
        return self.no_match()

      v = getattr(node, f)
      p = getattr(pattern, f)

      if self.is_wildcard(p):
        continue
      if isinstance(v, (list, tuple)):
        if not isinstance(p, (list, tuple)) or len(v) != len(p):
          return self.no_match()
        for v_item, p_item in zip(v, p):
          self.compare_and_visit(v_item, p_item)
      elif isinstance(v, (gast.AST, ast.AST)):
        if not isinstance(v, type(p)) and not isinstance(p, type(v)):
          return self.no_match()
        self.compare_and_visit(v, p)
      else:
        # Assume everything else is a value type.
        if v != p:
          return self.no_match()


def matches(node, pattern):
  # [FUNCTION DOC]
  # Tests whether syntax tree node matches pattern. pattern is a tree or source text (parsed
  # first); underscore names in it are wildcards. Returns true or false. Note that the match is
  # driven by the node's fields, and the top-level node types are not compared directly.
  """Basic pattern matcher for AST.

  The pattern may contain wildcards represented by the symbol '_'. A node
  matches a pattern if for every node in the tree, either there is a node of
  the same type in pattern, or a Name node with id='_'.

  Args:
    node: ast.AST
    pattern: ast.AST
  Returns:
    bool
  """
  if isinstance(pattern, str):
    pattern = parser.parse_str(pattern)

  matcher = PatternMatcher(pattern)
  matcher.visit(node)
  return matcher.matches


# TODO(mdan): Once we have error tracing, we may be able to just go to SSA.
def apply_to_single_assignments(targets, values, apply_fn):
  # [FUNCTION DOC]
  # Breaks a possibly unpacked assignment into single assignments and calls apply_fn(target,
  # value) for each. targets is one node or a list/tuple of nodes. For a tuple/list target, each
  # element is paired with the matching element of a tuple/list value, or with a subscript node of
  # the value at that position otherwise (recursing for nested targets). Plain targets are paired
  # with values as is. apply_fn results are ignored. Returns nothing.
  """Applies a function to each individual assignment.

  This function can process a possibly-unpacked (e.g. a, b = c, d) assignment.
  It tries to break down the unpacking if possible. In effect, it has the same
  effect as passing the assigned values in SSA form to apply_fn.

  Examples:

  The following will result in apply_fn(a, c), apply_fn(b, d):

      a, b = c, d

  The following will result in apply_fn(a, c[0]), apply_fn(b, c[1]):

      a, b = c

  The following will result in apply_fn(a, (b, c)):

      a = b, c

  It uses the visitor pattern to allow subclasses to process single
  assignments individually.

  Args:
    targets: Union[List[ast.AST, ...], Tuple[ast.AST, ...], ast.AST, should be
        used with the targets field of an ast.Assign node
    values: ast.AST
    apply_fn: Callable[[ast.AST, ast.AST], None], called with the
        respective nodes of each single assignment
  """
  if not isinstance(targets, (list, tuple)):
    targets = (targets,)
  for target in targets:
    if isinstance(target, (gast.Tuple, gast.List)):
      for i in range(len(target.elts)):
        target_el = target.elts[i]
        if isinstance(values, (gast.Tuple, gast.List)):
          value_el = values.elts[i]
        else:
          idx = parser.parse_expression(str(i))
          value_el = gast.Subscript(values, idx, ctx=gast.Load())
        apply_to_single_assignments(target_el, value_el, apply_fn)
    else:
      apply_fn(target, values)


def parallel_walk(node, other):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the second operand of the node-type validity check tests the first tree's node
  # for null instead of the second's, so a non-node, non-null item in the second tree opposite a
  # null item in the first is not rejected by that check (the class-name comparison still
  # catches most cases). Traversal is last-in-first-out, so pairs come in reverse child order.
  # Generator walking two trees (single nodes or lists/tuples of nodes) in parallel. Yields pairs
  # (node from first tree, node from second tree) before descending into children. Child fields
  # that are double-underscore named or null on either side are skipped; list fields must have
  # equal length. Raises a value error for nodes of differing type or inconsistent list lengths or
  # values.
  """Walks two ASTs in parallel.

  The two trees must have identical structure.

  Args:
    node: Union[ast.AST, Iterable[ast.AST]]
    other: Union[ast.AST, Iterable[ast.AST]]
  Yields:
    Tuple[ast.AST, ast.AST]
  Raises:
    ValueError: if the two trees don't have identical structure.
  """
  if isinstance(node, (list, tuple)):
    node_stack = list(node)
  else:
    node_stack = [node]

  if isinstance(other, (list, tuple)):
    other_stack = list(other)
  else:
    other_stack = [other]

  while node_stack and other_stack:
    assert len(node_stack) == len(other_stack)
    n = node_stack.pop()
    o = other_stack.pop()

    if ((not isinstance(n, (ast.AST, gast.AST, str)) and n is not None) or
        (not isinstance(o, (ast.AST, gast.AST, str)) and n is not None) or
        n.__class__.__name__ != o.__class__.__name__):
      raise ValueError('inconsistent nodes: {} ({}) and {} ({})'.format(
          n, n.__class__.__name__, o, o.__class__.__name__))

    yield n, o

    if isinstance(n, str):
      assert isinstance(o, str), 'The check above should have ensured this'
      continue
    if n is None:
      assert o is None, 'The check above should have ensured this'
      continue

    for f in n._fields:
      n_child = getattr(n, f, None)
      o_child = getattr(o, f, None)
      if f.startswith('__') or n_child is None or o_child is None:
        continue

      if isinstance(n_child, (list, tuple)):
        if (not isinstance(o_child, (list, tuple)) or
            len(n_child) != len(o_child)):
          raise ValueError(
              'inconsistent values for field {}: {} and {}'.format(
                  f, n_child, o_child))
        node_stack.extend(n_child)
        other_stack.extend(o_child)

      elif isinstance(n_child, (gast.AST, ast.AST)):
        node_stack.append(n_child)
        other_stack.append(o_child)

      elif n_child != o_child:
        raise ValueError(
            'inconsistent values for field {}: {} and {}'.format(
                f, n_child, o_child))
