# [MODULE DOC]
# Reaching function definitions analysis: determines, for each statement, which function
# definitions (and lambdas) may exist when it executes, accounting for closures (a nested
# function sees functions defined before it). Annotates statements with the set of defined
# function nodes. Exposes a definition holder, internal state and analyzer classes, a tree
# annotator and the resolve entry point. No import-time side effects.
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
"""An analysis that determines the reach of a function definition.

A function definition is said to reach a statement if that function may exist
(and therefore may be called) when that statement executes.
"""

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer


class Definition(object):
  # [CLASS DOC]
  # Holder for one function definition; attribute def_node is the function or lambda syntax node.
  # Note: not used by the analysis below, which tracks the nodes directly.
  """Definition objects describe a unique definition of a function."""

  def __init__(self, def_node):
    # [FUNCTION DOC]
    # Constructor. def_node is the function definition syntax node; it is stored as is.
    self.def_node = def_node


class _NodeState(object):
  # [CLASS DOC]
  # HELPER: dataflow lattice value used by the function-definitions Analyzer.
  # KNOWN ISSUE: the docstring describes a mapping from symbols to definitions, but the value is
  # actually a plain set of function definition nodes.
  # Value type holding the set of function/lambda nodes that may be defined at a point. Supports
  # equality, union with another state, and adding one node (returning a new state).
  """Abstraction for the state of the CFG walk for reaching definition analysis.

  This is a value type. Only implements the strictly necessary operators.

  Attributes:
    value: Dict[qual_names.QN, Set[Definition, ...]], the defined symbols and
        their possible definitions
  """

  def __init__(self, init_from=None):
    # [FUNCTION DOC]
    # Constructor. init_from is an optional collection of nodes; it is copied into a new set.
    # Nothing or empty gives an empty set.
    if init_from:
      self.value = set(init_from)
    else:
      self.value = set()

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Returns true if both states hold equal sets of nodes.
    return self.value == other.value

  def __ne__(self, other):
    # [FUNCTION DOC]
    # Returns true if the states hold different sets of nodes.
    return self.value != other.value

  def __or__(self, other):
    # [FUNCTION DOC]
    # Union of two states (other must be a state, asserted). Returns a new state; operands are
    # unchanged.
    assert isinstance(other, _NodeState)
    result = _NodeState(self.value)
    result.value.update(other.value)
    return result

  def __add__(self, value):
    # [FUNCTION DOC]
    # Returns a new state containing this state's nodes plus the given node. Original unchanged.
    result = _NodeState(self.value)
    result.value.add(value)
    return result

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a debug string with the object identity and the set contents.
    return 'NodeState[%s]=%s' % (id(self), repr(self.value))


class Analyzer(cfg.GraphVisitor):
  # [CLASS DOC]
  # Forward dataflow visitor over one function's control flow graph computing, for each node,
  # the function definitions that may exist at entry (in_) and exit (out). A function or lambda
  # node adds itself to its exit state. The graph entry additionally receives external
  # definitions (those visible from the enclosing function).
  """CFG visitor that determines reaching definitions at statement level."""

  def __init__(self, graph, external_defs):
    # [FUNCTION DOC]
    # Constructor. graph is the function's control flow graph; external_defs is a collection of
    # function nodes already defined when the function is entered (closure context). Stored for
    # use at the graph entry.
    super(Analyzer, self).__init__(graph)
    # This allows communicating that nodes have extra reaching definitions,
    # e.g. those that a function closes over.
    self.external_defs = external_defs

  def init_state(self, _):
    # [FUNCTION DOC]
    # Initial dataflow value for any node (argument ignored): a fresh empty state.
    return _NodeState()

  def visit_node(self, node):
    # [FUNCTION DOC]
    # Transfer function for one CFG node. Input state is the union of predecessors' outputs (at
    # the graph entry it also includes the external definitions). If the node's syntax node is a
    # function definition or lambda it is added to the output. Stores in/out states and returns
    # true if the output changed.
    # KNOWN ISSUE: for non-entry nodes the input starts from the node's previous output rather
    # than empty, so definitions never shrink between iterations (harmless for a monotone
    # analysis).
    prev_defs_out = self.out[node]

    if node is self.graph.entry:
      defs_in = _NodeState(self.external_defs)
    else:
      defs_in = prev_defs_out

    for n in node.prev:
      defs_in |= self.out[n]

    defs_out = defs_in
    if isinstance(node.ast_node, (gast.Lambda, gast.FunctionDef)):
      defs_out += node.ast_node

    self.in_[node] = defs_in
    self.out[node] = defs_out

    return prev_defs_out != defs_out


class TreeAnnotator(transformer.Base):
  # [CLASS DOC]
  # AST transformer that runs the Analyzer for each function or lambda (seeding it with the
  # definitions reaching the function node in its parent) and annotates every node having a graph
  # entry, plus extra loop test nodes, with the set of function nodes defined on entry. State:
  # graphs (mapping function nodes to graphs) and the current analyzer. Does not skip nodes.
  """AST visitor that annotates each symbol name with its reaching definitions.

  Simultaneously, the visitor runs the dataflow analysis on each function node,
  accounting for the effect of closures. For example:

    def foo():
      def f():
        pass
      def g():
        # `def f` reaches here
  """

  def __init__(self, source_info, graphs):
    # [FUNCTION DOC]
    # Constructor. source_info describes the source; graphs maps function and lambda nodes to
    # control flow graphs. Disables node skipping and starts with no current analyzer.
    super(TreeAnnotator, self).__init__(source_info)
    self.graphs = graphs
    self.allow_skips = False
    self.current_analyzer = None

  def _proces_function(self, node):
    # [FUNCTION DOC]
    # HELPER: shared implementation for function and lambda visits (name is misspelled).
    # Looks up the node's graph; if nested, takes the definitions reaching the node in the parent
    # analyzer as external definitions (else none). Runs a forward analysis, visits the children
    # with it as current, restores the parent analyzer, and returns the node.
    parent_analyzer = self.current_analyzer
    subgraph = self.graphs[node]

    if (self.current_analyzer is not None
        and node in self.current_analyzer.graph.index):
      cfg_node = self.current_analyzer.graph.index[node]
      defined_in = self.current_analyzer.in_[cfg_node].value
    else:
      defined_in = ()

    analyzer = Analyzer(subgraph, defined_in)
    analyzer.visit_forward()

    self.current_analyzer = analyzer
    node = self.generic_visit(node)
    self.current_analyzer = parent_analyzer
    return node

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Processes a function definition with its own analysis. Returns the node.
    return self._proces_function(node)

  def visit_Lambda(self, node):
    # [FUNCTION DOC]
    # Processes a lambda with its own analysis. Returns the node.
    return self._proces_function(node)

  def visit(self, node):
    # [FUNCTION DOC]
    # Generic dispatch override. Before visiting, if a current analyzer exists and the node has a
    # graph entry, annotates it with the function nodes defined on entry; likewise annotates the
    # extra loop test node, if the node carries one. Then delegates to the base visit and
    # returns its result.
    # This can happen before entering the top level function
    if (self.current_analyzer is not None
        and node in self.current_analyzer.graph.index):
      cfg_node = self.current_analyzer.graph.index[node]
      anno.setanno(node, anno.Static.DEFINED_FNS_IN,
                   self.current_analyzer.in_[cfg_node].value)

    extra_node = anno.getanno(node, anno.Basic.EXTRA_LOOP_TEST, default=None)
    if extra_node is not None:
      cfg_node = self.current_analyzer.graph.index[extra_node]
      anno.setanno(extra_node, anno.Static.DEFINED_FNS_IN,
                   self.current_analyzer.in_[cfg_node].value)

    return super(TreeAnnotator, self).visit(node)


def resolve(node, source_info, graphs):
  # [FUNCTION DOC]
  # Entry point. Annotates the tree in place with the function definitions reaching each
  # statement. node is the AST (module or function); source_info describes the source; graphs
  # maps function nodes to control flow graphs. Returns the annotated tree.
  # KNOWN ISSUE: the docstring speaks of reaching definitions for each symbol, but this module
  # tracks function definitions only.
  """Resolves reaching definitions for each symbol.

  Args:
    node: ast.AST
    source_info: transformer.SourceInfo
    graphs: Dict[ast.FunctionDef, cfg.Graph]
  Returns:
    ast.AST
  """
  visitor = TreeAnnotator(source_info, graphs)
  node = visitor.visit(node)
  return node
