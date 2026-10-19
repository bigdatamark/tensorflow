# [MODULE DOC]
# Records where generated autograph code came from. Defines immutable record types for source
# locations (LineLocation, Location) and per-node origin information (OriginInfo), a visitor
# (OriginResolver) and functions (resolve, resolve_entity) that annotate syntax-tree nodes with
# origin information based on the original source text, create_source_map which maps lines of
# generated code back to origin records by walking the original and re-parsed trees in parallel,
# and copy_origin which propagates an origin annotation to other nodes. Used for error rewriting
# and debugging of converted code. No import-time side effects beyond imports.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Container for origin source code information before AutoGraph compilation."""
import collections
import difflib
import io
import os
import tokenize

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.util import tf_inspect


class LineLocation(
    collections.namedtuple('LineLocation', ('filename', 'lineno'))):
  # [CLASS DOC]
  # Immutable record of a position in a file at line granularity: a file name and a 1-based line
  # number. Subclass of a named tuple with no extra behavior; used as the key type of source maps.
  """Similar to Location, but without column information.

  Attributes:
    filename: Text
    lineno: int, 1-based
  """
  pass


class Location(
    collections.namedtuple('Location', ('filename', 'lineno', 'col_offset'))):
  # [CLASS DOC]
  # Immutable record of a position in a file: file name, 1-based line number and column offset.
  # Subclass of a named tuple that adds a convenience property giving the line-only location.
  """Encodes code location information.

  Attributes:
    filename: Text
    lineno: int, 1-based
    col_offset: int
    line_loc: LineLocation
  """

  @property
  def line_loc(self):
    # [FUNCTION DOC]
    # Property returning a LineLocation built from this location's file name and line number
    # (dropping the column offset).
    return LineLocation(self.filename, self.lineno)


class OriginInfo(
    collections.namedtuple(
        'OriginInfo',
        ('loc', 'function_name', 'source_code_line', 'comment'))):
  # [CLASS DOC]
  # Immutable record describing the original (pre-conversion) source of a node: a Location, the
  # name of the enclosing function (may be null), the text of the source line and the trailing or
  # preceding comment on that line (may be null). Subclass of a named tuple that adds a
  # traceback-style frame conversion and a compact text form.
  """Container for information about the source code before conversion.

  Attributes:
    loc: Location
    function_name: Optional[Text]
    source_code_line: Text
    comment: Optional[Text]
  """

  def as_frame(self):
    # [FUNCTION DOC]
    # Returns a four-element tuple of file name, line number, function name and source line text,
    # the same shape as entries produced by the standard traceback extraction.
    """Returns a 4-tuple consistent with the return of traceback.extract_tb."""
    return (self.loc.filename, self.loc.lineno, self.function_name,
            self.source_code_line)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a short text form "file base name:line:column"; if the file name is empty or null,
    # uses a "no file" placeholder instead of the base name. Only the final path component is shown.
    if self.loc.filename:
      return '{}:{}:{}'.format(
          os.path.split(self.loc.filename)[1], self.loc.lineno,
          self.loc.col_offset)
    return '<no file>:{}:{}'.format(self.loc.lineno, self.loc.col_offset)


# TODO(mdan): This source map should be a class - easier to refer to.
def create_source_map(nodes, code, filepath):
  # [FUNCTION DOC]
  # Builds a mapping from line locations in generated code to origin information.
  # Inputs: nodes (one or more syntax trees already annotated with origin info), code (the
  # generated source text for those nodes), filepath (file name used for the generated code's
  # locations). The code is re-parsed without preamble and every re-parsed top-level node is
  # annotated with origin info through resolve. The original and re-parsed trees are then walked
  # in parallel; for each pair where both have origin annotations, the mapping key is the
  # re-parsed node's line location and the value is the original node's origin. Pairs lacking
  # either annotation are skipped (generated code may be unmapped). On key collision: if the
  # existing and new origins share a line location, the one with the larger line number is kept
  # (the existing one is kept when its line is greater or equal); otherwise, the one with the
  # smaller column offset is kept (existing kept when its offset is smaller or equal).
  # Raises a value error with a message containing a context diff of the pretty-printed original
  # and re-parsed trees when the parallel walk detects inconsistent trees. Returns the mapping.
  """Creates a source map between an annotated AST and the code it compiles to.

  Note: this function assumes nodes nodes, code and filepath correspond to the
  same code.

  Args:
    nodes: Iterable[ast.AST, ...], one or more AST modes.
    code: Text, the source code in which nodes are found.
    filepath: Text

  Returns:
    Dict[LineLocation, OriginInfo], mapping locations in code to locations
    indicated by origin annotations in node.
  """
  reparsed_nodes = parser.parse(code, preamble_len=0, single_node=False)
  for node in reparsed_nodes:
    resolve(node, code, filepath, node.lineno, node.col_offset)

  source_map = {}

  try:
    for before, after in ast_util.parallel_walk(nodes, reparsed_nodes):
      # Note: generated code might not be mapped back to its origin.
      # TODO(mdan): Generated code should always be mapped to something.
      origin_info = anno.getanno(before, anno.Basic.ORIGIN, default=None)
      final_info = anno.getanno(after, anno.Basic.ORIGIN, default=None)
      if origin_info is None or final_info is None:
        continue

      # Note: the keys are by line only, excluding the column offset.
      line_loc = LineLocation(final_info.loc.filename, final_info.loc.lineno)

      existing_origin = source_map.get(line_loc)
      if existing_origin is not None:
        # Overlaps may exist because of child nodes, but almost never to
        # different line locations. Exception make decorated functions, where
        # both lines are mapped to the same line in the AST.

        # Line overlaps: keep bottom node.
        if existing_origin.loc.line_loc == origin_info.loc.line_loc:
          if existing_origin.loc.lineno >= origin_info.loc.lineno:
            continue

        # In case of column overlaps, keep the leftmost node.
        if existing_origin.loc.col_offset <= origin_info.loc.col_offset:
          continue

      source_map[line_loc] = origin_info

  except ValueError as err:
    new_msg = 'Inconsistent ASTs detected. This is a bug. Cause: \n'
    new_msg += str(err)
    new_msg += 'Diff:\n'

    for n, rn in zip(nodes, reparsed_nodes):
      nodes_str = pretty_printer.fmt(n, color=False, noanno=True)
      reparsed_nodes_str = pretty_printer.fmt(rn, color=False, noanno=True)
      diff = difflib.context_diff(
          nodes_str.split('\n'),
          reparsed_nodes_str.split('\n'),
          fromfile='Original nodes',
          tofile='Reparsed nodes',
          n=7)
      diff = '\n'.join(diff)
      new_msg += diff + '\n'
    raise ValueError(new_msg)

  return source_map


class _Function:
  # [CLASS DOC]
  # HELPER: tiny record used by OriginResolver's function stack; holds only a function name.

  def __init__(self, name):
    # [FUNCTION DOC]
    # HELPER: stores the given function name (string) on the instance.
    self.name = name


class OriginResolver(gast.NodeVisitor):
  # [CLASS DOC]
  # Tree visitor that attaches an origin-info annotation (and a line number annotation) to every
  # node having a line number. It translates line numbers and column offsets of a node parsed
  # from a source snippet into absolute positions in the original file using an offset derived
  # from the root node and the known context position. Keeps a stack of enclosing function names
  # so each origin records its function. Created and driven by resolve.
  """Annotates an AST with additional source information like file name."""

  def __init__(self, root_node, source_lines, comments_map,
               context_lineno, context_col_offset,
               filepath):
    # [FUNCTION DOC]
    # Inputs: root_node (tree node being annotated), source_lines (list of text lines of the
    # snippet it was parsed from), comments_map (mapping from snippet line number to comment
    # text), context_lineno and context_col_offset (where the root node sits in the original
    # file), filepath (original file name).
    # Computes the line offset as context_lineno minus the line of the root's first decorator if
    # it has decorators with line numbers, else minus the root's own line; computes the column
    # offset as context_col_offset minus the root's column offset. Initializes an empty function stack.
    self._source_lines = source_lines
    self._comments_map = comments_map

    if (hasattr(root_node, 'decorator_list') and root_node.decorator_list and
        hasattr(root_node.decorator_list[0], 'lineno')):
      # Typical case: functions. The line number of the first decorator
      # is more accurate than the line number of the function itself in
      # 3.8+. In earlier versions they coincide.
      self._lineno_offset = context_lineno - root_node.decorator_list[0].lineno
    else:
      # Fall back to the line number of the root node.
      self._lineno_offset = context_lineno - root_node.lineno

    self._col_offset = context_col_offset - root_node.col_offset

    self._filepath = filepath

    self._function_stack = []

  def _absolute_lineno(self, lineno):
    # [FUNCTION DOC]
    # HELPER: position translation for OriginResolver.
    # Converts a line number in the snippet to the absolute line number by adding the stored offset.
    return lineno + self._lineno_offset

  def _absolute_col_offset(self, col_offset):
    # [FUNCTION DOC]
    # HELPER: position translation for OriginResolver.
    # Converts a snippet column offset to an absolute one by adding the stored offset; a null
    # column offset is mapped to zero.
    if col_offset is None:
      return 0
    return col_offset + self._col_offset

  def _attach_origin_info(self, node):
    # [FUNCTION DOC]
    # HELPER: annotation step for OriginResolver's visit.
    # If the node has no line number, does nothing. Otherwise builds an origin record from the
    # absolute location, the innermost enclosing function name (null at module level), the
    # snippet's source line and any comment on that line, and stores it on the node under the
    # origin annotation key; also stores the raw snippet line number under a line-number key.
    # Mutates the node's annotations; returns nothing.
    lineno = getattr(node, 'lineno', None)
    col_offset = getattr(node, 'col_offset', None)

    if lineno is None:
      return

    if self._function_stack:
      function_name = self._function_stack[-1].name
    else:
      function_name = None

    source_code_line = self._source_lines[lineno - 1]
    comment = self._comments_map.get(lineno)

    loc = Location(self._filepath, self._absolute_lineno(lineno),
                   self._absolute_col_offset(col_offset))
    origin = OriginInfo(loc, function_name, source_code_line, comment)
    anno.setanno(node, 'lineno', lineno)
    anno.setanno(node, anno.Basic.ORIGIN, origin)

  def visit(self, node):
    # [FUNCTION DOC]
    # Visitor entry for each node. If the node is a function definition, pushes its name on the
    # function stack; annotates the node with origin info, recursively visits its children, then
    # pops the stack if it pushed. Returns nothing (no transformed node is returned).
    entered_function = False
    if isinstance(node, gast.FunctionDef):
      entered_function = True
      self._function_stack.append(_Function(node.name))

    self._attach_origin_info(node)
    self.generic_visit(node)

    if entered_function:
      self._function_stack.pop()


def resolve(node, source, context_filepath, context_lineno, context_col_offset):
  # [FUNCTION DOC]
  # Annotates node (a tree) and its descendants with origin information.
  # Inputs: node, source (text the node was parsed from), context_filepath, context_lineno and
  # context_col_offset (where the node starts in the original file). The source is tokenized to
  # collect comments per line (comment text stripped of the marker). A tokenization error is
  # ignored when the node is a lambda (source recovery for lambdas can be garbage) and re-raised
  # otherwise. Then an OriginResolver visits the node. Mutates node annotations; returns nothing.
  """Adds origin information to an AST, based on the source it was loaded from.

  This allows us to map the original source code line numbers to generated
  source code.

  Note: the AST may be a part of a larger context (e.g. a function is part of
  a module that may contain other things). However, this function does not
  assume the source argument contains the entire context, nor that it contains
  only code corresponding to node itself. However, it assumes that node was
  parsed from the given source code.
  For this reason, two extra arguments are required, and they indicate the
  location of the node in the original context.

  Args:
    node: gast.AST, the AST to annotate.
    source: Text, the source code representing node.
    context_filepath: Text
    context_lineno: int
    context_col_offset: int
  """
  # TODO(mdan): Pull this to a separate utility.
  code_reader = io.StringIO(source)
  comments_map = {}
  try:
    for token in tokenize.generate_tokens(code_reader.readline):
      tok_type, tok_string, loc, _, _ = token
      srow, _ = loc
      if tok_type == tokenize.COMMENT:
        comments_map[srow] = tok_string.strip()[1:].strip()
  except tokenize.TokenError:
    if isinstance(node, gast.Lambda):
      # Source code resolution in older Python versions is brittle for
      # lambda functions, and may contain garbage.
      pass
    else:
      raise

  source_lines = source.split('\n')
  visitor = OriginResolver(node, source_lines, comments_map,
                           context_lineno, context_col_offset,
                           context_filepath)
  visitor.visit(node)


def resolve_entity(node, source, entity):
  # [FUNCTION DOC]
  # Like resolve, but derives the context from a live entity (function, class or method): the
  # file name and starting line come from the inspection utilities, and the column offset is
  # guessed as the count of leading whitespace characters of the entity's first source line
  # (unreliable with tabs). Then calls resolve with node and source. Returns nothing.
  """Like resolve, but extracts the context information from an entity."""
  lines, lineno = tf_inspect.getsourcelines(entity)
  filepath = tf_inspect.getsourcefile(entity)

  # Poor man's attempt at guessing the column offset: count the leading
  # whitespace. This might not work well with tabs.
  definition_line = lines[0]
  col_offset = len(definition_line) - len(definition_line.lstrip())

  resolve(node, source, filepath, lineno, col_offset)


def copy_origin(from_node, to_node):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring suggests copying from one node to another, but the origin is set
  # on to_node and on every descendant of it, not on from_node's tree structure.
  # Reads the origin annotation of from_node (null if absent; then nothing happens). to_node may
  # be a single node or a list/tuple of nodes; every node in each subtree (including the root)
  # receives that same origin annotation. Mutates annotations; returns nothing.
  """Copies the origin info from a node to another, recursively."""
  origin = anno.Basic.ORIGIN.of(from_node, default=None)
  if origin is None:
    return
  if not isinstance(to_node, (list, tuple)):
    to_node = (to_node,)
  for node in to_node:
    for n in gast.walk(node):
      anno.setanno(n, anno.Basic.ORIGIN, origin)
