# [MODULE DOC]
# Live variable analysis for AutoGraph. Provides Analyzer (a backward dataflow visitor over a
# control flow graph computing live-in/live-out symbol sets per graph node) and TreeAnnotator
# (a transformer that runs the analysis per function and attaches live-in/live-out symbol sets to
# statements as annotations). Entry point: resolve. Requires activity analysis, reaching function
# definition annotations and control flow graphs. No import-time side effects.
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
"""Live variable analysis.

See https://en.wikipedia.org/wiki/Live_variable_analysis for a definition of
the following idioms: live variable, live in, live out, which are used
throughout this file.

This analysis attaches the following:
 * symbols that are live at the exit of control flow statements
 * symbols that are live at the entry of control flow statements

Requires activity analysis.
"""

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import annos


class Analyzer(cfg.GraphVisitor):
  # [CLASS DOC]
  # Backward dataflow visitor over one control flow graph. Holds the include-annotations flag and
  # (from the base class) per-node in and out symbol sets. Run with the reverse visit.
  """CFG visitor that performs liveness analysis at statement level."""

  def __init__(self, graph, include_annotations):
    # [FUNCTION DOC]
    # Initializes the visitor with a control flow graph and a flag saying whether symbols used
    # only as type annotations count as reads.
    super(Analyzer, self).__init__(graph)
    self.include_annotations = include_annotations

  def init_state(self, _):
    # [FUNCTION DOC]
    # Returns the initial dataflow state for any node: an empty set of live symbols.
    return set()

  def lamba_check(self, fn_ast_node):
    # [FUNCTION DOC]
    # HELPER: returns true if the function node is a lambda, which is assumed to be used only
    # where defined. (The method name is misspelled in the source.)
    if isinstance(fn_ast_node, gast.Lambda):
      # Exception: lambda functions are assumed to be used only in the
      # place where they are defined, and not later.
      return True
    return False

  def visit_node(self, node):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when annotations are excluded, the set of reads is modified in place, which
    # also removes the annotation symbols from the node's recorded scope read set.
    # Computes live-in and live-out for one graph node. With a scope annotation: live-in is the
    # reads united with (union of successors' live-in minus modified and deleted symbols), plus
    # the closure (reads minus bound) of every function definition reaching the node, except
    # lambdas. Without a scope the node must be ignorable and live-in equals live-out. Stores
    # the sets and returns true if live-in changed from the previous value.
    prev_live_in = self.in_[node]

    if anno.hasanno(node.ast_node, anno.Static.SCOPE):
      node_scope = anno.getanno(node.ast_node, anno.Static.SCOPE)

      gen = node_scope.read
      if not self.include_annotations:
        gen -= node_scope.annotations
      # TODO(mdan): verify whether composites' parents need to be added.
      # E.g. whether x needs to be added if x.y is live. Theoretically the
      # activity analysis should have both so that wouldn't be needed.
      kill = node_scope.modified | node_scope.deleted

      live_out = set()
      for n in node.next:
        live_out |= self.in_[n]
      live_in = gen | (live_out - kill)

      reaching_functions = anno.getanno(
          node.ast_node, anno.Static.DEFINED_FNS_IN)
      for fn_ast_node in reaching_functions:
        if self.lamba_check(fn_ast_node):
          continue
        fn_scope = anno.getanno(fn_ast_node, annos.NodeAnno.ARGS_AND_BODY_SCOPE)
        # Any closure of a reaching function definition is conservatively
        # considered live.
        live_in |= (fn_scope.read - fn_scope.bound)

    else:
      assert self.can_ignore(node), (node.ast_node, node)

      live_out = set()
      for n in node.next:
        live_out |= self.in_[n]
      live_in = live_out

    self.in_[node] = live_in
    self.out[node] = live_out

    # TODO(mdan): Move this to the superclass?
    return prev_live_in != live_in


class TreeAnnotator(transformer.Base):
  # [CLASS DOC]
  # Transformer that runs liveness analysis on every function or lambda in a tree, using the
  # supplied mapping from function nodes to control flow graphs, and attaches live-in and
  # live-out annotations to statements. Holds the include-annotations flag, the graphs and the
  # analyzer of the function currently being visited.
  """Runs liveness analysis on each of the functions defined in the AST.

  If a function defined other local functions, those will have separate CFGs.
  However, dataflow analysis needs to tie up these CFGs to properly emulate the
  effect of closures. In the case of liveness, the parent function's live
  variables must account for the variables that are live at the entry of each
  subfunction. For example:

    def foo():
      # baz is live from here on
      def bar():
        print(baz)

  This analyzer runs liveness analysis on each individual function, accounting
  for the effect above.
  """

  def __init__(self, source_info, graphs, include_annotations):
    # [FUNCTION DOC]
    # Initializes with source info, the mapping of function nodes to graphs and the
    # include-annotations flag; disables node skipping and starts with no current analyzer.
    super(TreeAnnotator, self).__init__(source_info)
    self.include_annotations = include_annotations
    self.allow_skips = False
    self.graphs = graphs
    self.current_analyzer = None

  def visit(self, node):
    # [FUNCTION DOC]
    # Visits a node normally, then, if inside an analyzed function and the node is a statement
    # present in the graph, annotates it with the frozen live-in set. Returns the node.
    node = super(TreeAnnotator, self).visit(node)
    if (self.current_analyzer is not None and
        isinstance(node, gast.stmt) and
        node in self.current_analyzer.graph.index):
      cfg_node = self.current_analyzer.graph.index[node]
      anno.setanno(node, anno.Static.LIVE_VARS_IN,
                   frozenset(self.current_analyzer.in_[cfg_node]))
    return node

  def _analyze_function(self, node, is_lambda):
    # [FUNCTION DOC]
    # HELPER: runs a reverse liveness analysis on the function's graph, makes it current while
    # visiting children, then restores the outer analyzer. The lambda flag is unused. Returns node.
    parent_analyzer = self.current_analyzer

    analyzer = Analyzer(self.graphs[node], self.include_annotations)
    analyzer.visit_reverse()
    self.current_analyzer = analyzer
    node = self.generic_visit(node)

    self.current_analyzer = parent_analyzer
    return node

  def visit_Lambda(self, node):
    # [FUNCTION DOC]
    # Visitor for lambdas: analyzes it as a function.
    return self._analyze_function(node, is_lambda=True)

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Visitor for function definitions: analyzes it as a function.
    return self._analyze_function(node, is_lambda=False)

  def _block_statement_live_out(self, node):
    # [FUNCTION DOC]
    # HELPER: annotates a block statement with the union of live-in sets of its statement-level
    # successors in the graph as live-out (frozen). Returns the node.
    successors = self.current_analyzer.graph.stmt_next[node]
    stmt_live_out = set()
    for s in successors:
      stmt_live_out.update(self.current_analyzer.in_[s])
    anno.setanno(node, anno.Static.LIVE_VARS_OUT, frozenset(stmt_live_out))
    return node

  def _block_statement_live_in(self, node, entry_node):
    # [FUNCTION DOC]
    # HELPER: annotates a block statement with the live-in set of its entry node (such as the
    # test or first body statement): from the graph if present, otherwise from the entry node's
    # existing live-in annotation (asserts it exists). Returns the node.
    if entry_node in self.current_analyzer.graph.index:
      cfg_node = self.current_analyzer.graph.index[entry_node]
      stmt_live_in = frozenset(self.current_analyzer.in_[cfg_node])
    else:
      assert anno.hasanno(entry_node, anno.Static.LIVE_VARS_IN), (
          'If not matching a CFG node, must be a block statement:'
          ' {}'.format(entry_node))
      stmt_live_in = anno.getanno(entry_node, anno.Static.LIVE_VARS_IN)
    anno.setanno(node, anno.Static.LIVE_VARS_IN, stmt_live_in)
    return node

  def visit_If(self, node):
    # [FUNCTION DOC]
    # Visitor for if statements: after visiting children, sets live-out and sets live-in from the
    # test. Returns the node.
    node = self.generic_visit(node)
    node = self._block_statement_live_out(node)
    return self._block_statement_live_in(node, node.test)

  def visit_For(self, node):
    # [FUNCTION DOC]
    # Visitor for for-loops: after visiting children, sets live-out and sets live-in from the
    # iterable. Returns the node.
    node = self.generic_visit(node)
    node = self._block_statement_live_out(node)
    return self._block_statement_live_in(node, node.iter)

  def visit_While(self, node):
    # [FUNCTION DOC]
    # Visitor for while loops: after visiting children, sets live-out and sets live-in from the
    # test. Returns the node.
    node = self.generic_visit(node)
    node = self._block_statement_live_out(node)
    return self._block_statement_live_in(node, node.test)

  def visit_Try(self, node):
    # [FUNCTION DOC]
    # Visitor for try statements: after visiting children, sets live-out and sets live-in from the
    # first statement of the body. Returns the node.
    node = self.generic_visit(node)
    node = self._block_statement_live_out(node)
    return self._block_statement_live_in(node, node.body[0])

  def visit_ExceptHandler(self, node):
    # [FUNCTION DOC]
    # Visitor for except handlers: after visiting children, sets live-out and sets live-in from the
    # first statement of the handler body. Returns the node.
    node = self.generic_visit(node)
    node = self._block_statement_live_out(node)
    return self._block_statement_live_in(node, node.body[0])

  def visit_With(self, node):
    # [FUNCTION DOC]
    # Visitor for with statements: after visiting children, sets live-in from the first with item
    # (no live-out is set). Returns the node.
    node = self.generic_visit(node)
    return self._block_statement_live_in(node, node.items[0])

  def visit_Expr(self, node):
    # [FUNCTION DOC]
    # Visitor for expression statements: after visiting children, annotates the statement with the
    # live-out set of its graph node (frozen). Requires the statement to be in the graph. Returns
    # the node.
    node = self.generic_visit(node)
    cfg_node = self.current_analyzer.graph.index[node]
    anno.setanno(node, anno.Static.LIVE_VARS_OUT,
                 frozenset(self.current_analyzer.out[cfg_node]))
    return node


# TODO(mdan): Investigate the possibility of removing include_annotations.
def resolve(node, source_info, graphs, include_annotations=True):
  # [FUNCTION DOC]
  # Entry point: runs the liveness tree annotator over a syntax tree. Inputs: the tree, source
  # info, mapping from function nodes to control flow graphs, and flag (default true) to count
  # type-annotation symbols as reads. Returns the annotated tree; mutates it.
  """Resolves the live symbols at the exit of control flow statements.

  Args:
    node: ast.AST
    source_info: transformer.SourceInfo
    graphs: Dict[ast.FunctionDef, cfg.Graph]
    include_annotations: Bool, whether type annotations should be included in
      the analysis.
  Returns:
    ast.AST
  """
  node = TreeAnnotator(source_info, graphs, include_annotations).visit(node)
  return node
