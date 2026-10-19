# [MODULE DOC]
# Reaching-definitions dataflow analysis for autograph's static analysis. Walks the control flow
# graphs of each function, computing for every statement the set of Definition objects that may
# reach it, and annotates every name node with its possible definitions. Also annotates
# control-flow statements with the set of symbols defined on entry. Exposes the Definition class,
# the internal state/analyzer/annotator classes, and the resolve entry point. Requires prior
# activity (scope) analysis and CFG construction. No import-time side effects.
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
"""Reaching definition analysis.

This analysis attaches a set of a Definition objects to each symbol, one
for each distinct definition that may reach it. The Definition objects are
mutable and may be used by subsequent analyses to further annotate data like
static type and value information.
The analysis also attaches the set of the symbols defined at the entry of
control flow statements.

Requires activity analysis.
"""

import weakref

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer


class Definition(object):
  # [CLASS DOC]
  # Identity object representing one unique definition (binding) of a variable. Compared by object
  # identity, so each definition site must create exactly one. Attributes: param_of (weak
  # reference to the parameter AST node if this definition is a function parameter, else
  # nothing) and directives (mapping for annotations added by later analyses). Subclasses can be
  # supplied through a factory passed to resolve.
  """Definition objects describe a unique definition of a variable.

  Subclasses of this may be used by passing an appropriate factory function to
  resolve.

  Attributes:
    param_of: Optional[ast.AST]
    directives: Dict, optional definition annotations
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Takes no arguments. Sets param_of to nothing (not a parameter definition) and
    # directives to an empty mapping. Returns nothing.
    self.param_of = None
    self.directives = {}

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a string with the class name and the object identity number in brackets.
    return '%s[%d]' % (self.__class__.__name__, id(self))


class _NodeState(object):
  # [CLASS DOC]
  # HELPER: dataflow lattice value used by the reaching-definitions Analyzer.
  # Value type mapping each defined symbol to the set of Definition objects that may reach a
  # point. Can be built empty, copied from another state (deep copies the sets), or from a
  # mapping of symbol to a single definition. Supports equality, union (merge at join points)
  # and subtraction of a set of symbols (kill).
  """Abstraction for the state of the CFG walk for reaching definition analysis.

  This is a value type. Only implements the strictly necessary operators.

  Attributes:
    value: Dict[qual_names.QN, Set[Definition, ...]], the defined symbols and
        their possible definitions
  """

  def __init__(self, init_from=None):
    # [FUNCTION DOC]
    # Constructor. init_from is optional: nothing or empty gives an empty state; another state
    # gives a copy with copied sets; a mapping from symbol to a single definition gives a state
    # where each symbol has a one-element set. Any other non-empty type fails an assertion.
    if init_from:
      if isinstance(init_from, _NodeState):
        self.value = {
            s: set(other_infos) for s, other_infos in init_from.value.items()
        }
      elif isinstance(init_from, dict):
        self.value = {s: set((init_from[s],)) for s in init_from}
      else:
        assert False, init_from
    else:
      self.value = {}

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Returns true if both states have the same set of symbols and each symbol maps to an equal
    # set of definitions.
    if frozenset(self.value.keys()) != frozenset(other.value.keys()):
      return False
    ret = all(self.value[s] == other.value[s] for s in self.value)
    return ret

  def __ne__(self, other):
    # [FUNCTION DOC]
    # Returns the negation of equality.
    return not self.__eq__(other)

  def __or__(self, other):
    # [FUNCTION DOC]
    # Union of two states. other must be a state (asserted). Returns a new state in which each
    # symbol maps to the union of the definition sets from both operands. Neither operand is
    # modified.
    assert isinstance(other, _NodeState)
    result = _NodeState(self)
    for s, other_infos in other.value.items():
      if s in result.value:
        result.value[s].update(other_infos)
      else:
        result.value[s] = set(other_infos)
    return result

  def __sub__(self, other):
    # [FUNCTION DOC]
    # Kill operation. other must be a set of symbols (asserted). Returns a new state with those
    # symbols removed (missing ones ignored). The original is not modified.
    assert isinstance(other, set)
    result = _NodeState(self)
    for s in other:
      result.value.pop(s, None)
    return result

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a debug string with the object identity and the contents of the symbol mapping.
    return 'NodeState[%s]=%s' % (id(self), repr(self.value))


class Analyzer(cfg.GraphVisitor):
  # [CLASS DOC]
  # Forward dataflow visitor over one function's control flow graph. For each graph node it
  # computes the definitions reaching its entry (in_) and exit (out), as _NodeState values.
  # Keeps gen_map, a cache of the definitions generated by each node so that definition objects
  # are created only once per node (their identity is used for equality and fixpoint checks).
  """CFG visitor that determines reaching definitions at statement level."""

  def __init__(self, graph, definition_factory):
    # [FUNCTION DOC]
    # Constructor. graph is the control flow graph of one function; definition_factory is a
    # callable with no arguments returning a new definition object. Initializes the base visitor
    # and an empty gen_map. Returns nothing.
    self._definition_factory = definition_factory
    super(Analyzer, self).__init__(graph)
    self.gen_map = {}

  def init_state(self, _):
    # [FUNCTION DOC]
    # Initial dataflow value for any graph node (argument ignored): a fresh empty state.
    return _NodeState()

  def visit_node(self, node):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the lines fetching the generated state and computing the output are
    # duplicated, which is redundant but harmless.
    # Transfer function for one CFG node. Input state is the union of the predecessors' outputs.
    # If the node's AST carries scope annotations, generated definitions are: one per bound or
    # global symbol not deleted, plus one per parameter (marked with a weak reference to the
    # parameter node); they are created once and cached in gen_map. Output is generated
    # definitions merged with the input minus symbols modified or deleted. Nodes without scope
    # annotations must be ignorable (asserted) and pass state through unchanged. Stores the in
    # and out states and returns true if the output changed from the previous iteration.
    prev_defs_out = self.out[node]

    defs_in = _NodeState()
    for n in node.prev:
      defs_in |= self.out[n]

    if anno.hasanno(node.ast_node, anno.Static.SCOPE):
      node_scope = anno.getanno(node.ast_node, anno.Static.SCOPE)
      # The definition objects created by each node must be singletons because
      # their ids are used in equality checks.
      if node not in self.gen_map:
        node_symbols = {}
        # Every binding operation (assign, nonlocal, global, etc.) counts as a
        # definition, with the exception of del, which only deletes without
        # creating a new variable.
        newly_defined = ((node_scope.bound | node_scope.globals) -
                         node_scope.deleted)
        for s in newly_defined:
          def_ = self._definition_factory()
          node_symbols[s] = def_
        # Every param receives a definition. Params are not necessarily
        # considered as "modified".
        for s, p in node_scope.params.items():
          def_ = self._definition_factory()
          def_.param_of = weakref.ref(p)
          node_symbols[s] = def_
        self.gen_map[node] = _NodeState(node_symbols)

      gen = self.gen_map[node]
      kill = node_scope.modified | node_scope.deleted
      defs_out = gen | (defs_in - kill)

      gen = self.gen_map[node]
      defs_out = gen | (defs_in - kill)

    else:
      assert self.can_ignore(node), (node.ast_node, node)
      defs_out = defs_in

    self.in_[node] = defs_in
    self.out[node] = defs_out

    return prev_defs_out != defs_out


class TreeAnnotator(transformer.Base):
  # [CLASS DOC]
  # AST transformer that runs the Analyzer on every function and annotates name nodes with
  # their reaching definitions (loads use definitions at statement entry, stores use those at
  # exit), and annotates if/for/while/try/except nodes with the symbols defined on entry. State:
  # graphs (mapping from function nodes to their CFGs), definition_factory, the current
  # analyzer and current CFG node. Does not skip nodes.
  """AST visitor that annotates each symbol name with its reaching definitions.

  Simultaneously, the visitor runs the dataflow analysis on each function node,
  accounting for the effect of closures. For example:

    def foo():
      bar = 1
      def baz():
        # bar = 1 reaches here
  """

  def __init__(self, source_info, graphs, definition_factory):
    # [FUNCTION DOC]
    # Constructor. source_info describes the source being processed; graphs maps function
    # definition nodes to their control flow graphs; definition_factory creates definition
    # objects. Disables node skipping and clears current analyzer and CFG node.
    super(TreeAnnotator, self).__init__(source_info)
    self.allow_skips = False
    self.definition_factory = definition_factory
    self.graphs = graphs
    self.current_analyzer = None
    self.current_cfg_node = None

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Runs a forward reaching-definitions analysis on this function's graph, then visits its
    # arguments and body with that analyzer as current (nested functions get their own), and
    # restores the parent analyzer. Returns the same function node, annotated.
    parent_analyzer = self.current_analyzer
    subgraph = self.graphs[node]

    analyzer = Analyzer(subgraph, self.definition_factory)
    analyzer.visit_forward()

    # Recursively process any remaining subfunctions.
    self.current_analyzer = analyzer
    node.args = self.visit(node.args)
    node.body = self.visit_block(node.body)
    self.current_analyzer = parent_analyzer

    return node

  def visit_Name(self, node):
    # [FUNCTION DOC]
    # Annotates a name node with a tuple of definitions: for loads, those in the analyzer's input
    # state at the current CFG node; otherwise those in its output state. Names outside any
    # function (e.g. in class bodies) are returned unannotated. Fails an assertion if a name
    # occurs outside any statement. Returns the node.
    if self.current_analyzer is None:
      # Names may appear outside function defs - for example in class
      # definitions.
      return node

    analyzer = self.current_analyzer
    cfg_node = self.current_cfg_node

    assert cfg_node is not None, ('name node, %s, outside of any statement?'
                                  % node.id)

    qn = anno.getanno(node, anno.Basic.QN)
    if isinstance(node.ctx, gast.Load):
      anno.setanno(node, anno.Static.DEFINITIONS,
                   tuple(analyzer.in_[cfg_node].value.get(qn, ())))
    else:
      anno.setanno(node, anno.Static.DEFINITIONS,
                   tuple(analyzer.out[cfg_node].value.get(qn, ())))

    return node

  def _aggregate_predecessors_defined_in(self, node):
    # [FUNCTION DOC]
    # HELPER: used by the visit methods of control-flow statements.
    # Collects the symbols defined at exit of all statement-level predecessors of the node in the
    # current graph and annotates the node with them as a frozen set of variables defined on
    # entry. Returns nothing.
    preds = self.current_analyzer.graph.stmt_prev[node]
    node_defined_in = set()
    for p in preds:
      node_defined_in |= set(self.current_analyzer.out[p].value.keys())
    anno.setanno(node, anno.Static.DEFINED_VARS_IN, frozenset(node_defined_in))

  def visit_If(self, node):
    # [FUNCTION DOC]
    # Annotates the if statement with symbols defined on entry, then visits children. Returns node.
    self._aggregate_predecessors_defined_in(node)
    return self.generic_visit(node)

  def visit_For(self, node):
    # [FUNCTION DOC]
    # Annotates the for statement with symbols defined on entry. Because the loop target is not
    # a separate statement in the graph, visits the target with the CFG node of the iterated
    # expression as current (so target definitions are the ones produced at the iterator), then
    # visits iter, body and else-body. Returns the node.
    self._aggregate_predecessors_defined_in(node)

    # Manually accounting for the shortcoming described in
    # cfg.AstToCfg.visit_For.
    parent = self.current_cfg_node
    self.current_cfg_node = self.current_analyzer.graph.index[node.iter]
    node.target = self.visit(node.target)
    self.current_cfg_node = parent

    node.iter = self.visit(node.iter)
    node.body = self.visit_block(node.body)
    node.orelse = self.visit_block(node.orelse)

    return node

  def visit_While(self, node):
    # [FUNCTION DOC]
    # Annotates the while statement with symbols defined on entry, then visits children. Returns node.
    self._aggregate_predecessors_defined_in(node)
    return self.generic_visit(node)

  def visit_Try(self, node):
    # [FUNCTION DOC]
    # Annotates the try statement with symbols defined on entry, then visits children. Returns node.
    self._aggregate_predecessors_defined_in(node)
    return self.generic_visit(node)

  def visit_ExceptHandler(self, node):
    # [FUNCTION DOC]
    # Annotates the except handler with symbols defined on entry and visits only its body (the
    # exception type and name are not visited; a TODO notes this). Returns the node.
    self._aggregate_predecessors_defined_in(node)
    # TODO(mdan): Also track the exception type / name symbols.
    node.body = self.visit_block(node.body)
    return node

  def visit(self, node):
    # [FUNCTION DOC]
    # Generic dispatch override. If inside a function and the node has its own entry in the
    # current graph, makes that graph node current while visiting it, restoring the previous one
    # afterwards. Returns the visited (possibly replaced) node.
    parent = self.current_cfg_node

    if (self.current_analyzer is not None and
        node in self.current_analyzer.graph.index):
      self.current_cfg_node = self.current_analyzer.graph.index[node]
    node = super(TreeAnnotator, self).visit(node)

    self.current_cfg_node = parent
    return node


def resolve(node, source_info, graphs, definition_factory=Definition):
  # [FUNCTION DOC]
  # Entry point. Runs the reaching-definitions annotator over a parsed tree. node is the AST to
  # analyze (typically a module or function); source_info describes its source; graphs maps
  # function nodes to control flow graphs; definition_factory (optional, defaults to the
  # Definition class) creates definition objects. Annotates names with their definitions and
  # control-flow statements with symbols defined on entry, in place. Returns the annotated tree.
  """Resolves reaching definitions for each symbol.

  Args:
    node: ast.AST
    source_info: transformer.SourceInfo
    graphs: Dict[ast.FunctionDef, cfg.Graph]
    definition_factory: Callable[[], Definition]
  Returns:
    ast.AST
  """
  visitor = TreeAnnotator(source_info, graphs, definition_factory)
  node = visitor.visit(node)
  return node
