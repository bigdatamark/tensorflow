# [MODULE DOC]
# Unit tests for the autograph control flow graph (cfg) module. Two test classes: one checks the
# generic graph visitor (forward and reverse traversal coverage) and one checks that the graph
# built from small parsed Python functions has the expected nodes, predecessor/successor edges,
# statement-level edges, entry node and exit nodes. Covers straight-line code, if/else, while,
# for (with else, break, continue, return), try/except/finally, raise, with, lambdas,
# comprehensions, class definitions and imports. Defines no flags or registrations at import time.
# Note: the small nested functions inside each test are intentionally left without comment
# blocks, because the tests assert on source line numbers of the parsed function and extra
# comment lines would shift them.
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
"""Tests for cfg module."""

import gast

from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.platform import test


class CountingVisitor(cfg.GraphVisitor):
  # [CLASS DOC]
  # HELPER: test visitor used by GraphVisitorTest.
  # Graph visitor subclass that counts how many times each underlying syntax-tree node is
  # visited. Holds a mapping (counts) from syntax-tree node to visit count. Its visit step always
  # reports "no change", so every reachable node is visited exactly once.

  def __init__(self, graph):
    # [FUNCTION DOC]
    # Constructor. Takes the control flow graph to traverse, passes it to the base visitor and
    # starts with an empty counts mapping. Returns nothing.
    super(CountingVisitor, self).__init__(graph)
    self.counts = {}

  def init_state(self, _):
    # [FUNCTION DOC]
    # HELPER: state initializer required by the graph visitor protocol.
    # Ignores the node argument and returns nothing/null as the initial state for every node.
    return None

  def visit_node(self, node):
    # [FUNCTION DOC]
    # Records a visit: increments the count stored for the node's syntax-tree node (starting
    # from zero). Always returns false so that the traversal does not revisit nodes.
    self.counts[node.ast_node] = self.counts.get(node.ast_node, 0) + 1
    return False  # visit only once


class GraphVisitorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the generic graph visitor in the cfg module, using CountingVisitor on graphs built
  # from small parsed functions. Checks which nodes are reached in forward and reverse order.

  def _build_cfg(self, fn):
    # [FUNCTION DOC]
    # HELPER: shared fixture builder for the visitor tests.
    # Parses the given Python function into a syntax tree (no future features) and builds its
    # control flow graphs. Returns a pair: mapping of graphs, and the parsed function node.
    node, _ = parser.parse_entity(fn, future_features=())
    cfgs = cfg.build(node)
    return cfgs, node

  def test_basic_coverage_forward(self):
    # [FUNCTION DOC]
    # Scenario: a function with a while loop containing an assignment, a break and a statement
    # after the break, followed by an assignment. A counting visitor traverses forward.
    # Expected: arguments, loop test, assignment, break and the trailing statement are each
    # visited once; the statement after break is unreachable and never visited.

    def test_fn(a):
      while a > 0:
        a = 1
        break
        return a  # pylint:disable=unreachable
      a = 2

    graphs, node = self._build_cfg(test_fn)
    graph, = graphs.values()
    visitor = CountingVisitor(graph)
    visitor.visit_forward()

    self.assertEqual(visitor.counts[node.args], 1)
    self.assertEqual(visitor.counts[node.body[0].test], 1)
    self.assertEqual(visitor.counts[node.body[0].body[0]], 1)
    self.assertEqual(visitor.counts[node.body[0].body[1]], 1)
    # The return node should be unreachable in forward direction.
    self.assertNotIn(node.body[0].body[2], visitor.counts)
    self.assertEqual(visitor.counts[node.body[1]], 1)

  def test_basic_coverage_reverse(self):
    # [FUNCTION DOC]
    # Same function as the forward test, traversed in reverse. Expected: every node, including
    # the statement after break that is unreachable going forward, is visited exactly once.

    def test_fn(a):
      while a > 0:
        a = 1
        break
        return a  # pylint:disable=unreachable
      a = 2

    graphs, node = self._build_cfg(test_fn)
    graph, = graphs.values()
    visitor = CountingVisitor(graph)
    visitor.visit_reverse()

    self.assertEqual(visitor.counts[node.args], 1)
    self.assertEqual(visitor.counts[node.body[0].test], 1)
    self.assertEqual(visitor.counts[node.body[0].body[0]], 1)
    self.assertEqual(visitor.counts[node.body[0].body[1]], 1)
    self.assertEqual(visitor.counts[node.body[0].body[2]], 1)
    self.assertEqual(visitor.counts[node.body[1]], 1)


class AstToCfgTest(test.TestCase):
  # [CLASS DOC]
  # Tests that the cfg builder turns parsed functions into the expected control flow graphs.
  # Provides assertion helpers that compare graph nodes (by their printed form), predecessor and
  # successor sets, statement-level edges, the entry node and the set of exit nodes. Each test
  # defines a small function, builds its graph(s) and asserts on those properties.

  def _build_cfg(self, fn):
    # [FUNCTION DOC]
    # HELPER: shared fixture builder. Parses the given function (no future features), builds the
    # control flow graphs and returns the mapping from syntax node (function or lambda) to graph.
    node, _ = parser.parse_entity(fn, future_features=())
    cfgs = cfg.build(node)
    return cfgs

  def _repr_set(self, node_set):
    # [FUNCTION DOC]
    # HELPER: returns an immutable set of the printed forms of the given graph nodes.
    # Appears unused in this file.
    return frozenset(repr(n) for n in node_set)

  def _as_set(self, elements):
    # [FUNCTION DOC]
    # HELPER: normalizes an expected-edges specification into an immutable set of strings.
    # Nothing/null gives an empty set, a single string gives a one-element set, any other
    # collection is converted to a set.
    if elements is None:
      return frozenset()
    elif isinstance(elements, str):
      return frozenset((elements,))
    else:
      return frozenset(elements)

  def assertGraphMatches(self, graph, edges):
    # [FUNCTION DOC]
    # HELPER: custom assertion shared by the tests.
    # Takes a graph and a list of triples (expected predecessors, node printed form, expected
    # successors). For each triple, searches the graph's nodes for one with that printed form
    # whose predecessor and successor sets match exactly. Fails the test with a message showing
    # the graph if no node matches. Returns nothing.
    """Tests whether the CFG contains the specified edges."""
    for prev, node_repr, next_ in edges:
      matched = False
      for cfg_node in graph.index.values():
        if repr(cfg_node) == node_repr:
          if (self._as_set(prev) == frozenset(map(repr, cfg_node.prev)) and
              self._as_set(next_) == frozenset(map(repr, cfg_node.next))):
            matched = True
            break
      if not matched:
        self.fail(
            'match failed for node "%s" in graph:\n%s' % (node_repr, graph))

  def assertGraphEnds(self, graph, entry_repr, exit_reprs):
    # [FUNCTION DOC]
    # HELPER: custom assertion. Checks that the graph's entry node prints as entry_repr and that
    # the set of printed forms of its exit nodes equals exit_reprs (any collection of strings).
    # Fails the test otherwise. Returns nothing.
    """Tests whether the CFG has the specified entry and exits."""
    self.assertEqual(repr(graph.entry), entry_repr)
    self.assertSetEqual(frozenset(map(repr, graph.exit)), frozenset(exit_reprs))

  def assertStatementEdges(self, graph, edges):
    # [FUNCTION DOC]
    # HELPER: custom assertion for statement-level edges.
    # Takes a graph and triples (expected previous node forms, statement label of the form
    # "<node type>:<line number>", expected next node forms). First asserts the graph's
    # statement-next and statement-previous mappings have the same statements. Then finds the
    # statement with the matching label and compares its previous and next node sets. Fails with
    # the list of partial matches if none matches. Returns nothing.
    """Tests whether the CFG contains the specified statement edges."""
    for prev_node_reprs, node_repr, next_node_reprs in edges:
      matched = False
      partial_matches = []
      self.assertSetEqual(
          frozenset(graph.stmt_next.keys()), frozenset(graph.stmt_prev.keys()))
      for stmt_ast_node in graph.stmt_next:
        ast_repr = '%s:%s' % (stmt_ast_node.__class__.__name__,
                              stmt_ast_node.lineno)
        if ast_repr == node_repr:
          actual_next = frozenset(map(repr, graph.stmt_next[stmt_ast_node]))
          actual_prev = frozenset(map(repr, graph.stmt_prev[stmt_ast_node]))
          partial_matches.append((actual_prev, node_repr, actual_next))
          if (self._as_set(prev_node_reprs) == actual_prev and
              self._as_set(next_node_reprs) == actual_next):
            matched = True
            break
      if not matched:
        self.fail('edges mismatch for %s: %s' % (node_repr, partial_matches))

  def test_straightline(self):
    # [FUNCTION DOC]
    # Scenario: sequential assignments ending in a bare return. Expects a chain of nodes from the
    # argument node to the return, and the return as the only exit.

    def test_fn(a):
      a += 1
      a = 2
      a = 3
      return

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (None, 'a', 'a += 1'),
            ('a += 1', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', 'return'),
            ('a = 3', 'return', None),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('return',))

  def test_straightline_no_return(self):
    # [FUNCTION DOC]
    # Scenario: two statements with no return. Expects a simple chain from the arguments to the
    # last statement, which is the only exit.

    def test_fn(a, b):
      a = b + 1
      a += max(a)

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (None, 'a, b', 'a = (b + 1)'),
            ('a = (b + 1)', 'a += max(a)', None),
        ),
    )
    self.assertGraphEnds(graph, 'a, b', ('a += max(a)',))

  def test_unreachable_code(self):
    # [FUNCTION DOC]
    # Scenario: a statement placed after a return. Expects the unreachable statement to be a
    # node with no predecessors and no successors, and both it and the return to be exits.

    def test_fn(a):
      return
      a += 1  # pylint:disable=unreachable

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (None, 'a', 'return'),
            ('a', 'return', None),
            (None, 'a += 1', None),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('return', 'a += 1'))

  def test_if_straightline(self):
    # [FUNCTION DOC]
    # Scenario: if/else with one assignment per branch. Expects the condition to branch to both
    # assignments, the if statement edge to have no successor, and both branches to be exits.

    def test_fn(a):
      if a > 0:
        a = 1
      else:
        a += -1

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (None, 'a', '(a > 0)'),
            ('(a > 0)', 'a = 1', None),
            ('(a > 0)', 'a += (- 1)', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'If:2', None),),
    )
    self.assertGraphEnds(graph, 'a', ('a = 1', 'a += (- 1)'))

  def test_branch_nested(self):
    # [FUNCTION DOC]
    # Scenario: nested if/else in both branches of an outer if/else. Expects the correct tree of
    # condition-to-branch edges, statement edges for the three ifs, and four leaf exits.

    def test_fn(a):
      if a > 0:
        if a > 1:
          a = 1
        else:
          a = 2
      else:
        if a > 2:
          a = 3
        else:
          a = 4

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (None, 'a', '(a > 0)'),
            ('a', '(a > 0)', ('(a > 1)', '(a > 2)')),
            ('(a > 0)', '(a > 1)', ('a = 1', 'a = 2')),
            ('(a > 1)', 'a = 1', None),
            ('(a > 1)', 'a = 2', None),
            ('(a > 0)', '(a > 2)', ('a = 3', 'a = 4')),
            ('(a > 2)', 'a = 3', None),
            ('(a > 2)', 'a = 4', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'If:2', None),
            ('(a > 0)', 'If:3', None),
            ('(a > 0)', 'If:8', None),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 1', 'a = 2', 'a = 3', 'a = 4'))

  def test_branch_straightline_unbalanced(self):
    # [FUNCTION DOC]
    # Scenario: an if with no else. Expects the condition to be an exit as well as the body
    # (the false path falls through), and the condition to lead to the body.

    def test_fn(a):
      if a > 0:
        a = 1

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (None, 'a', '(a > 0)'),
            ('a', '(a > 0)', 'a = 1'),
            ('(a > 0)', 'a = 1', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'If:2', None),),
    )
    self.assertGraphEnds(graph, 'a', ('(a > 0)', 'a = 1'))

  def test_branch_return(self):
    # [FUNCTION DOC]
    # Scenario: one branch returns, the other assigns, then a trailing assignment. Expects the
    # return to have no successor, the other branch to flow to the trailing statement, and the
    # trailing statement and the return to be the exits.

    def test_fn(a):
      if a > 0:
        return
      else:
        a = 1
      a = 2

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', '(a > 0)', ('return', 'a = 1')),
            ('(a > 0)', 'a = 1', 'a = 2'),
            ('(a > 0)', 'return', None),
            ('a = 1', 'a = 2', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'If:2', 'a = 2'),),
    )
    self.assertGraphEnds(graph, 'a', ('a = 2', 'return'))

  def test_branch_raise(self):
    # [FUNCTION DOC]
    # Same as the branch-return test but one branch raises. Expects the raise to have no
    # successor and to be an exit along with the trailing assignment.

    def test_fn(a):
      if a > 0:
        raise a
      else:
        a = 1
      a = 2

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', '(a > 0)', ('raise a', 'a = 1')),
            ('(a > 0)', 'a = 1', 'a = 2'),
            ('(a > 0)', 'raise a', None),
            ('a = 1', 'a = 2', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'If:2', 'a = 2'),),
    )
    self.assertGraphEnds(graph, 'a', ('a = 2', 'raise a'))

  def test_branch_return_minimal(self):
    # [FUNCTION DOC]
    # Scenario: an if whose only body statement is a return, no else. Expects both the condition
    # and the return to be exits, and the if statement to have no successor.

    def test_fn(a):
      if a > 0:
        return

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', '(a > 0)', 'return'),
            ('(a > 0)', 'return', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'If:2', None),),
    )
    self.assertGraphEnds(graph, 'a', ('(a > 0)', 'return'))

  def test_while_straightline(self):
    # [FUNCTION DOC]
    # Scenario: a while loop with one body statement followed by an assignment. Expects the loop
    # test to have the entry and body as predecessors, a back edge from body to test, and the
    # test leading to the following statement, which is the sole exit.

    def test_fn(a):
      while a > 0:
        a = 1
      a = 2

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), '(a > 0)', ('a = 1', 'a = 2')),
            ('(a > 0)', 'a = 1', '(a > 0)'),
            ('(a > 0)', 'a = 2', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'While:2', 'a = 2'),),
    )
    self.assertGraphEnds(graph, 'a', ('a = 2',))

  def test_while_else_straightline(self):
    # [FUNCTION DOC]
    # Scenario: while loop with an else clause and a trailing statement. Expects the loop test to
    # lead to the body and to the else statement, which then flows to the trailing statement.

    def test_fn(a):
      while a > 0:
        a = 1
      else:  # pylint:disable=useless-else-on-loop
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), '(a > 0)', ('a = 1', 'a = 2')),
            ('(a > 0)', 'a = 1', '(a > 0)'),
            ('(a > 0)', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'While:2', 'a = 3'),),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3',))

  def test_while_else_continue(self):
    # [FUNCTION DOC]
    # Scenario: while/else where an inner if either continues or assigns. Expects continue to
    # have an edge back to the loop test, and the loop test to have the continue among its
    # predecessors; the else statement leads to the trailing statement.

    def test_fn(a):
      while a > 0:
        if a > 1:
          continue
        else:
          a = 0
        a = 1
      else:  # pylint:disable=useless-else-on-loop
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'continue', 'a = 1'), '(a > 0)', ('(a > 1)', 'a = 2')),
            ('(a > 0)', '(a > 1)', ('continue', 'a = 0')),
            ('(a > 1)', 'continue', '(a > 0)'),
            ('a = 0', 'a = 1', '(a > 0)'),
            ('(a > 0)', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'While:2', 'a = 3'),
            ('(a > 0)', 'If:3', ('a = 1', '(a > 0)')),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3',))

  def test_while_else_break(self):
    # [FUNCTION DOC]
    # Scenario: while/else with a break inside an if. Expects break to jump past the else clause
    # directly to the trailing statement, which has both break and the else body as predecessors.

    def test_fn(a):
      while a > 0:
        if a > 1:
          break
        a = 1
      else:
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), '(a > 0)', ('(a > 1)', 'a = 2')),
            ('(a > 0)', '(a > 1)', ('break', 'a = 1')),
            ('(a > 1)', 'break', 'a = 3'),
            ('(a > 1)', 'a = 1', '(a > 0)'),
            ('(a > 0)', 'a = 2', 'a = 3'),
            (('break', 'a = 2'), 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'While:2', 'a = 3'),
            ('(a > 0)', 'If:3', ('a = 1', 'a = 3')),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3',))

  def test_while_else_return(self):
    # [FUNCTION DOC]
    # Scenario: while/else with a return inside an if. Expects the return to have no successor
    # and to be an exit along with the trailing statement.

    def test_fn(a):
      while a > 0:
        if a > 1:
          return
        a = 1
      else:  # pylint:disable=useless-else-on-loop
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), '(a > 0)', ('(a > 1)', 'a = 2')),
            ('(a > 0)', '(a > 1)', ('return', 'a = 1')),
            ('(a > 1)', 'return', None),
            ('(a > 1)', 'a = 1', '(a > 0)'),
            ('(a > 0)', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'While:2', 'a = 3'),
            ('(a > 0)', 'If:3', 'a = 1'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3', 'return'))

  def test_while_nested_straightline(self):
    # [FUNCTION DOC]
    # Scenario: a while loop nested in a while loop. Expects inner and outer back edges and
    # statement edges for both loops; the statement after the outer loop is the sole exit.

    def test_fn(a):
      while a > 0:
        while a > 1:
          a = 1
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 2'), '(a > 0)', ('(a > 1)', 'a = 3')),
            (('(a > 0)', 'a = 1'), '(a > 1)', ('a = 1', 'a = 2')),
            ('(a > 1)', 'a = 1', '(a > 1)'),
            ('(a > 1)', 'a = 2', '(a > 0)'),
            ('(a > 0)', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'While:2', 'a = 3'),
            ('(a > 0)', 'While:3', 'a = 2'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3',))

  def test_while_nested_continue(self):
    # [FUNCTION DOC]
    # Scenario: continue inside an if inside an inner while. Expects continue to loop back to the
    # inner loop test (not the outer one), and the if statement edge to reflect that.

    def test_fn(a):
      while a > 0:
        while a > 1:
          if a > 3:
            continue
          a = 1
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 2'), '(a > 0)', ('(a > 1)', 'a = 3')),
            (('(a > 0)', 'continue', 'a = 1'), '(a > 1)', ('(a > 3)', 'a = 2')),
            ('(a > 1)', '(a > 3)', ('continue', 'a = 1')),
            ('(a > 3)', 'continue', '(a > 1)'),
            ('(a > 3)', 'a = 1', '(a > 1)'),
            ('(a > 1)', 'a = 2', '(a > 0)'),
            ('(a > 0)', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'While:2', 'a = 3'),
            ('(a > 0)', 'While:3', 'a = 2'),
            ('(a > 1)', 'If:4', ('a = 1', '(a > 1)')),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3',))

  def test_while_nested_break(self):
    # [FUNCTION DOC]
    # Scenario: break inside an if inside an inner while. Expects break to leave only the inner
    # loop, flowing to the statement after the inner loop inside the outer loop.

    def test_fn(a):
      while a > 0:
        while a > 1:
          if a > 2:
            break
          a = 1
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(graph, (
        (('a', 'a = 2'), '(a > 0)', ('(a > 1)', 'a = 3')),
        (('(a > 0)', 'a = 1'), '(a > 1)', ('(a > 2)', 'a = 2')),
        ('(a > 1)', '(a > 2)', ('break', 'a = 1')),
        ('(a > 2)', 'break', 'a = 2'),
        ('(a > 2)', 'a = 1', '(a > 1)'),
        (('(a > 1)', 'break'), 'a = 2', '(a > 0)'),
        ('(a > 0)', 'a = 3', None),
    ))
    self.assertStatementEdges(
        graph,
        (
            ('a', 'While:2', 'a = 3'),
            ('(a > 0)', 'While:3', 'a = 2'),
            ('(a > 1)', 'If:4', ('a = 1', 'a = 2')),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3',))

  def test_for_straightline(self):
    # [FUNCTION DOC]
    # Scenario: a for loop over a range with one body statement, then an assignment. Expects the
    # iterator expression node to act as the loop head with a back edge from the body and an edge
    # to the trailing statement, which is the sole exit.

    def test_fn(a):
      for a in range(0, a):
        a = 1
      a = 2

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), 'range(0, a)', ('a = 1', 'a = 2')),
            ('range(0, a)', 'a = 1', 'range(0, a)'),
            ('range(0, a)', 'a = 2', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'For:2', 'a = 2'),),
    )
    self.assertGraphEnds(graph, 'a', ('a = 2',))

  def test_for_else_straightline(self):
    # [FUNCTION DOC]
    # Scenario: for loop with an else clause and a trailing statement. Expects the loop head to
    # lead to the body and to the else statement, which flows to the trailing statement.

    def test_fn(a):
      for a in range(0, a):
        a = 1
      else:  # pylint:disable=useless-else-on-loop
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), 'range(0, a)', ('a = 1', 'a = 2')),
            ('range(0, a)', 'a = 1', 'range(0, a)'),
            ('range(0, a)', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (('a', 'For:2', 'a = 3'),),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3',))

  def test_for_else_continue(self):
    # [FUNCTION DOC]
    # Scenario: for/else where an inner if either continues or assigns. Expects continue to jump
    # back to the loop head and the head to have continue among its predecessors.

    def test_fn(a):
      for a in range(0, a):
        if a > 1:
          continue
        else:
          a = 0
        a = 1
      else:  # pylint:disable=useless-else-on-loop
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'continue', 'a = 1'), 'range(0, a)', ('(a > 1)', 'a = 2')),
            ('range(0, a)', '(a > 1)', ('continue', 'a = 0')),
            ('(a > 1)', 'continue', 'range(0, a)'),
            ('(a > 1)', 'a = 0', 'a = 1'),
            ('a = 0', 'a = 1', 'range(0, a)'),
            ('range(0, a)', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'For:2', 'a = 3'),
            ('range(0, a)', 'If:3', ('a = 1', 'range(0, a)')),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3',))

  def test_for_else_break(self):
    # [FUNCTION DOC]
    # Scenario: for/else with a break inside an if. Expects break to skip the else clause and go
    # straight to the trailing statement.

    def test_fn(a):
      for a in range(0, a):
        if a > 1:
          break
        a = 1
      else:
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), 'range(0, a)', ('(a > 1)', 'a = 2')),
            ('range(0, a)', '(a > 1)', ('break', 'a = 1')),
            ('(a > 1)', 'break', 'a = 3'),
            ('(a > 1)', 'a = 1', 'range(0, a)'),
            ('range(0, a)', 'a = 2', 'a = 3'),
            (('break', 'a = 2'), 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'For:2', 'a = 3'),
            ('range(0, a)', 'If:3', ('a = 1', 'a = 3')),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3',))

  def test_for_else_return(self):
    # [FUNCTION DOC]
    # Scenario: for/else with a return inside an if. Expects the return to have no successor and
    # to be an exit together with the trailing statement.

    def test_fn(a):
      for a in range(0, a):
        if a > 1:
          return
        a = 1
      else:  # pylint:disable=useless-else-on-loop
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), 'range(0, a)', ('(a > 1)', 'a = 2')),
            ('range(0, a)', '(a > 1)', ('return', 'a = 1')),
            ('(a > 1)', 'return', None),
            ('(a > 1)', 'a = 1', 'range(0, a)'),
            ('range(0, a)', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'For:2', 'a = 3'),
            ('range(0, a)', 'If:3', 'a = 1'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3', 'return'))

  def test_for_nested_straightline(self):
    # [FUNCTION DOC]
    # Scenario: a for loop nested in a for loop. Expects separate loop heads with proper back
    # edges and statement edges; the statement after the outer loop is the sole exit.

    def test_fn(a):
      for a in range(0, a):
        for b in range(1, a):
          b += 1
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 2'), 'range(0, a)', ('range(1, a)', 'a = 3')),
            (('range(0, a)', 'b += 1'), 'range(1, a)', ('b += 1', 'a = 2')),
            ('range(1, a)', 'b += 1', 'range(1, a)'),
            ('range(1, a)', 'a = 2', 'range(0, a)'),
            ('range(0, a)', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'For:2', 'a = 3'),
            ('range(0, a)', 'For:3', 'a = 2'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3',))

  def test_for_nested_continue(self):
    # [FUNCTION DOC]
    # Scenario: continue inside an if inside an inner for loop. Expects continue to return to the
    # inner loop head only.

    def test_fn(a):
      for a in range(0, a):
        for b in range(1, a):
          if a > 3:
            continue
          b += 1
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 2'), 'range(0, a)', ('range(1, a)', 'a = 3')),
            (('range(0, a)', 'continue', 'b += 1'), 'range(1, a)',
             ('(a > 3)', 'a = 2')),
            ('range(1, a)', '(a > 3)', ('continue', 'b += 1')),
            ('(a > 3)', 'continue', 'range(1, a)'),
            ('(a > 3)', 'b += 1', 'range(1, a)'),
            ('range(1, a)', 'a = 2', 'range(0, a)'),
            ('range(0, a)', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'For:2', 'a = 3'),
            ('range(0, a)', 'For:3', 'a = 2'),
            ('range(1, a)', 'If:4', ('b += 1', 'range(1, a)')),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3',))

  def test_for_nested_break(self):
    # [FUNCTION DOC]
    # Scenario: break inside an if inside an inner for loop. Expects break to exit only the inner
    # loop, flowing to the statement that follows it in the outer loop body.

    def test_fn(a):
      for a in range(0, a):
        for b in range(1, a):
          if a > 2:
            break
          b += 1
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 2'), 'range(0, a)', ('range(1, a)', 'a = 3')),
            (('range(0, a)', 'b += 1'), 'range(1, a)', ('(a > 2)', 'a = 2')),
            ('range(1, a)', '(a > 2)', ('break', 'b += 1')),
            ('(a > 2)', 'break', 'a = 2'),
            ('(a > 2)', 'b += 1', 'range(1, a)'),
            (('range(1, a)', 'break'), 'a = 2', 'range(0, a)'),
            ('range(0, a)', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'For:2', 'a = 3'),
            ('range(0, a)', 'For:3', 'a = 2'),
            ('range(1, a)', 'If:4', ('b += 1', 'a = 2')),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3',))

  def test_complex(self):
    # [FUNCTION DOC]
    # Scenario: a function mixing a while loop, a for/else with break, continue and return, nested
    # ifs and a trailing for loop with a return. Expects the full edge set, statement edges for
    # each compound statement, and exits at the final assignment and both returns.

    def test_fn(a):
      b = 0
      while a > 0:
        for b in range(0, a):
          if a > 2:
            break
          if a > 3:
            if a > 4:
              continue
            else:
              max(a)
              break
          b += 1
        else:  # for b in range(0, a):
          return a
        a = 2
      for a in range(1, a):
        return b
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('b = 0', 'a = 2'), '(a > 0)', ('range(0, a)', 'range(1, a)')),
            (
                ('(a > 0)', 'continue', 'b += 1'),
                'range(0, a)',
                ('(a > 2)', 'return a'),
            ),
            ('range(0, a)', '(a > 2)', ('(a > 3)', 'break')),
            ('(a > 2)', 'break', 'a = 2'),
            ('(a > 2)', '(a > 3)', ('(a > 4)', 'b += 1')),
            ('(a > 3)', '(a > 4)', ('continue', 'max(a)')),
            ('(a > 4)', 'max(a)', 'break'),
            ('max(a)', 'break', 'a = 2'),
            ('(a > 4)', 'continue', 'range(0, a)'),
            ('(a > 3)', 'b += 1', 'range(0, a)'),
            ('range(0, a)', 'return a', None),
            ('break', 'a = 2', '(a > 0)'),
            ('(a > 0)', 'range(1, a)', ('return b', 'a = 3')),
            ('range(1, a)', 'return b', None),
            ('range(1, a)', 'a = 3', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('b = 0', 'While:3', 'range(1, a)'),
            ('(a > 0)', 'For:4', 'a = 2'),
            ('range(0, a)', 'If:5', ('(a > 3)', 'a = 2')),
            ('(a > 2)', 'If:7', ('b += 1', 'a = 2', 'range(0, a)')),
            ('(a > 3)', 'If:8', ('a = 2', 'range(0, a)')),
            ('(a > 0)', 'For:17', 'a = 3'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3', 'return a', 'return b'))

  def test_finally_straightline(self):
    # [FUNCTION DOC]
    # Scenario: try body then finally block then a trailing statement. Expects a straight chain
    # body, finally, trailing statement; the trailing statement is the sole exit.

    def test_fn(a):
      try:
        a += 1
      finally:
        a = 2
      a = 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'a += 1', 'a = 2'),
            ('a += 1', 'a = 2', 'a = 3'),
            ('a = 2', 'a = 3', None),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 3',))

  def test_return_finally(self):
    # [FUNCTION DOC]
    # Scenario: a return inside try with a finally block, and a statement after the try. Expects
    # the return to flow into the finally block, the code after the try to be unreachable, and the
    # finally block and unreachable statement to be the exits (finally runs after the return).

    def test_fn(a):
      try:
        return a
      finally:
        a = 1
      a = 2

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'return a', 'a = 1'),
            ('return a', 'a = 1', None),
            (None, 'a = 2', None),
        ),
    )
    # Note, `a = 1` executes after `return a`.
    self.assertGraphEnds(graph, 'a', ('a = 2', 'a = 1'))

  def test_break_finally(self):
    # [FUNCTION DOC]
    # Scenario: a break inside try/finally within a while loop. Expects break to flow into the
    # finally block; the loop test and the finally block are the exits.

    def test_fn(a):
      while a > 0:
        try:
          break
        finally:
          a = 1

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', '(a > 0)', 'break'),
            ('(a > 0)', 'break', 'a = 1'),
            ('break', 'a = 1', None),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('(a > 0)', 'a = 1'))

  def test_continue_finally(self):
    # [FUNCTION DOC]
    # Scenario: a continue inside try/finally within a while loop. Expects continue to flow into
    # the finally block, which then returns to the loop test; the loop test is the only exit.

    def test_fn(a):
      while a > 0:
        try:
          continue
        finally:
          a = 1

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            (('a', 'a = 1'), '(a > 0)', 'continue'),
            ('(a > 0)', 'continue', 'a = 1'),
            ('continue', 'a = 1', '(a > 0)'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('(a > 0)',))

  def test_with_straightline(self):
    # [FUNCTION DOC]
    # Scenario: a with statement whose body assigns and returns. Expects the context expression
    # to be a node preceding the body statements in a straight chain; the return is the exit.

    def test_fn(a):
      with max(a) as b:
        a = 0
        return b

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'max(a)', 'a = 0'),
            ('max(a)', 'a = 0', 'return b'),
            ('a = 0', 'return b', None),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('return b',))

  def test_lambda_basic(self):
    # [FUNCTION DOC]
    # Scenario: an assignment of a lambda followed by a return. Expects two graphs: one for the
    # function, where the lambda expression is its own node before the assignment, and one for
    # the lambda, with its argument entry node leading to its body expression, the sole exit.

    def test_fn(a):
      a = lambda b: a + b
      return a

    graphs = self._build_cfg(test_fn)
    for k, v in graphs.items():
      if isinstance(k, gast.Lambda):
        lam_graph = v
      else:
        fn_graph = v

    self.assertGraphMatches(
        fn_graph,
        (
            ('a', '(lambda b: (a + b))', 'a = (lambda b: (a + b))'),
            ('(lambda b: (a + b))', 'a = (lambda b: (a + b))', 'return a'),
            ('a = (lambda b: (a + b))', 'return a', None),
        ),
    )
    self.assertGraphEnds(fn_graph, 'a', ('return a',))
    self.assertGraphMatches(
        lam_graph,
        (
            ('b', '(a + b)', None),
        ),
    )
    self.assertGraphEnds(lam_graph, 'b', ('(a + b)',))

  def test_lambda_in_return(self):
    # [FUNCTION DOC]
    # Scenario: a function that returns a lambda. Expects separate graphs for the function and the
    # lambda, with the lambda expression node preceding the return in the function graph.

    def test_fn(a):
      return lambda b: a + b

    graphs = self._build_cfg(test_fn)
    for k, v in graphs.items():
      if isinstance(k, gast.Lambda):
        lam_graph = v
      else:
        fn_graph = v

    self.assertGraphMatches(
        fn_graph,
        (
            ('a', '(lambda b: (a + b))', 'return (lambda b: (a + b))'),
            ('(lambda b: (a + b))', 'return (lambda b: (a + b))', None),
        ),
    )
    self.assertGraphEnds(fn_graph, 'a', ('return (lambda b: (a + b))',))
    self.assertGraphMatches(
        lam_graph,
        (
            ('b', '(a + b)', None),
        ),
    )
    self.assertGraphEnds(lam_graph, 'b', ('(a + b)',))

  def test_lambda_in_while_loop_test(self):
    # [FUNCTION DOC]
    # Scenario: a lambda called inside a while loop test. Expects the lambda node to precede the
    # loop test node, the loop body to loop back to the test, and a separate lambda graph.

    def test_fn(a):
      while (lambda b: a + b)(a):
        pass

    graphs = self._build_cfg(test_fn)
    for k, v in graphs.items():
      if isinstance(k, gast.Lambda):
        lam_graph = v
      else:
        fn_graph = v

    self.assertGraphMatches(
        fn_graph,
        (
            ('a', '(lambda b: (a + b))', '(lambda b: (a + b))(a)'),
            (('(lambda b: (a + b))', 'pass'), '(lambda b: (a + b))(a)', 'pass'),
            ('(lambda b: (a + b))(a)', 'pass', '(lambda b: (a + b))(a)'),
        ),
    )
    self.assertGraphEnds(fn_graph, 'a', ('(lambda b: (a + b))(a)',))
    self.assertGraphMatches(
        lam_graph,
        (
            ('b', '(a + b)', None),
        ),
    )
    self.assertGraphEnds(lam_graph, 'b', ('(a + b)',))

  def test_lambda_in_for_loop_test(self):
    # [FUNCTION DOC]
    # Scenario: a lambda called in the iterable expression of a for loop. Expects the same graph
    # shape as the while-test case: lambda node before the iterable node, loop back edge from the
    # body, and a separate lambda graph.

    def test_fn(a):
      for _ in (lambda b: a + b)(a):
        pass

    graphs = self._build_cfg(test_fn)
    for k, v in graphs.items():
      if isinstance(k, gast.Lambda):
        lam_graph = v
      else:
        fn_graph = v

    self.assertGraphMatches(
        fn_graph,
        (
            ('a', '(lambda b: (a + b))', '(lambda b: (a + b))(a)'),
            (('(lambda b: (a + b))', 'pass'), '(lambda b: (a + b))(a)', 'pass'),
            ('(lambda b: (a + b))(a)', 'pass', '(lambda b: (a + b))(a)'),
        ),
    )
    self.assertGraphEnds(fn_graph, 'a', ('(lambda b: (a + b))(a)',))
    self.assertGraphMatches(
        lam_graph,
        (
            ('b', '(a + b)', None),
        ),
    )
    self.assertGraphEnds(lam_graph, 'b', ('(a + b)',))

  def test_pass(self):
    # [FUNCTION DOC]
    # Scenario: a function whose body is only a pass statement. Expects the pass node after the
    # arguments and to be the sole exit.

    def test_fn(a):  # pylint:disable=unused-argument
      pass

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'pass', None),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('pass',))

  def test_try_finally(self):
    # [FUNCTION DOC]
    # Scenario: try body, finally block, then a return. Expects a straight chain and a statement
    # edge from the try statement to the return.

    def test_fn(a):
      try:
        a = 1
      finally:
        a = 2
      return a

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'a = 1', 'a = 2'),
            ('a = 1', 'a = 2', 'return a'),
            ('a = 2', 'return a', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'Try:2', 'return a'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('return a',))

  def test_try_except_single_bare(self):
    # [FUNCTION DOC]
    # Scenario: a try body of two statements with a bare except handler, then a return. Expects
    # the handler to be reachable from the last body statement and both to lead to the return.

    def test_fn(a):
      try:
        a = 1
        a = 2
      except:  # pylint:disable=bare-except
        a = 3
      return a

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'a = 1', 'a = 2'),
            ('a = 2', 'a = 3', 'return a'),
            (('a = 2', 'a = 3'), 'return a', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'Try:2', 'return a'),
            ('a = 2', 'ExceptHandler:5', 'return a'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('return a',))

  def test_try_except_single(self):
    # [FUNCTION DOC]
    # Same as the bare-except case but with a handler naming an exception type. Expects identical
    # graph edges and statement edges.

    def test_fn(a):
      try:
        a = 1
        a = 2
      except Exception1:  # pylint:disable=undefined-variable
        a = 3
      return a

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'a = 1', 'a = 2'),
            ('a = 2', 'a = 3', 'return a'),
            (('a = 2', 'a = 3'), 'return a', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'Try:2', 'return a'),
            ('a = 2', 'ExceptHandler:5', 'return a'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('return a',))

  def test_try_except_single_aliased(self):
    # [FUNCTION DOC]
    # Scenario: a one-statement try body with a handler that binds the exception to a name.
    # Expects the body to lead to both the handler body and the return, with the handler also
    # leading to the return.

    def test_fn(a):
      try:
        a = 1
      except Exception1 as e:  # pylint:disable=undefined-variable,unused-variable
        a = 2
      return a

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'a = 1', ('a = 2', 'return a')),
            (('a = 1', 'a = 2'), 'return a', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'Try:2', 'return a'),
            ('a = 1', 'ExceptHandler:4', 'return a'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('return a',))

  def test_try_except_single_tuple_aliased(self):
    # [FUNCTION DOC]
    # Same as the aliased case but the handler names a tuple of exception types. Expects the
    # same graph and statement edges.

    def test_fn(a):
      try:
        a = 1
      except (Exception1, Exception2) as e:  # pylint:disable=undefined-variable,unused-variable
        a = 2
      return a

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'a = 1', ('a = 2', 'return a')),
            (('a = 1', 'a = 2'), 'return a', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'Try:2', 'return a'),
            ('a = 1', 'ExceptHandler:4', 'return a'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('return a',))

  def test_try_except_multiple(self):
    # [FUNCTION DOC]
    # Scenario: a try body with two except handlers and a following return. Expects the body to
    # lead to both handlers and the return, and each handler to lead to the return.

    def test_fn(a):
      try:
        a = 1
      except Exception1:  # pylint:disable=undefined-variable
        a = 2
      except Exception2:  # pylint:disable=undefined-variable
        a = 3
      return a

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'a = 1', ('a = 2', 'a = 3', 'return a')),
            (('a = 1', 'a = 2', 'a = 3'), 'return a', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'Try:2', 'return a'),
            ('a = 1', 'ExceptHandler:4', 'return a'),
            ('a = 1', 'ExceptHandler:6', 'return a'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('return a',))

  def test_try_except_finally(self):
    # [FUNCTION DOC]
    # Scenario: try with two handlers and a finally block, then a return. Expects the body and
    # handlers to flow into the finally block, which flows to the return.

    def test_fn(a):
      try:
        a = 1
      except Exception1:  # pylint:disable=undefined-variable
        a = 2
      except Exception2:  # pylint:disable=undefined-variable
        a = 3
      finally:
        a = 4
      return a

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'a = 1', ('a = 2', 'a = 3', 'a = 4')),
            (('a = 1', 'a = 2', 'a = 3'), 'a = 4', 'return a'),
            ('a = 4', 'return a', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'Try:2', 'return a'),
            ('a = 1', 'ExceptHandler:4', 'a = 4'),
            ('a = 1', 'ExceptHandler:6', 'a = 4'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('return a',))

  def test_try_in_if(self):
    # [FUNCTION DOC]
    # Scenario: an if/else inside a try body with one handler and a trailing assignment. Expects
    # both branches to lead to the handler and the trailing statement, and statement edges for the
    # try, the if and the handler.

    def test_fn(a):
      try:
        if a > 0:
          a = 1
        else:
          a = 2
      except Exception1:  # pylint:disable=undefined-variable
        a = 3
      a = 4

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', '(a > 0)', ('a = 1', 'a = 2')),
            ('(a > 0)', 'a = 1', ('a = 3', 'a = 4')),
            ('(a > 0)', 'a = 2', ('a = 3', 'a = 4')),
            (('a = 1', 'a = 2'), 'a = 3', 'a = 4'),
            (('a = 1', 'a = 2', 'a = 3'), 'a = 4', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a', 'Try:2', 'a = 4'),
            ('a', 'If:3', ('a = 3', 'a = 4')),
            (('a = 1', 'a = 2'), 'ExceptHandler:7', 'a = 4'),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('a = 4',))

  def test_try_in_if_all_branches_exit(self):
    # [FUNCTION DOC]
    # Scenario: inside a try, one branch raises and the other returns; the handler returns.
    # Expects the raise to lead to the handler, the return to have no successor, and the exits to
    # be both returns and the raise.

    def test_fn(a, b):
      try:
        if a > 0:
          raise b
        else:
          return 0
      except b:
        return 1

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a, b', '(a > 0)', ('raise b', 'return 0')),
            ('(a > 0)', 'raise b', 'return 1'),
            ('(a > 0)', 'return 0', None),
            ('raise b', 'return 1', None),
        ),
    )
    self.assertStatementEdges(
        graph,
        (
            ('a, b', 'Try:2', None),
            ('a, b', 'If:3', 'return 1'),
            ('raise b', 'ExceptHandler:7', None),
        ),
    )
    self.assertGraphEnds(graph, 'a, b', ('return 0', 'return 1', 'raise b'))

  def test_raise_exits(self):
    # [FUNCTION DOC]
    # Scenario: a raise followed by an unreachable return. Expects the raise to have no
    # successor, the return to be disconnected, and both to be exits.

    def test_fn(a, b):
      raise b
      return a  # pylint:disable=unreachable

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a, b', 'raise b', None),
            (None, 'return a', None),
        ),
    )
    self.assertGraphEnds(graph, 'a, b', ('raise b', 'return a'))

  def test_raise_triggers_enclosing_finally(self):
    # [FUNCTION DOC]
    # Scenario: a raise inside nested try/finally blocks. Expects the raise to flow into the inner
    # finally, which flows into the outer finally; returns that cannot be reached stay linked only
    # to their finally blocks. Exits are the final return and the outer finally block.

    def test_fn(a):
      try:
        try:
          raise a
          return 1  # pylint:disable=unreachable
        finally:
          b = 1
        return 2
      finally:
        b = 2
      return b

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'raise a', 'b = 1'),
            (('raise a', 'return 1'), 'b = 1', 'b = 2'),
            (None, 'return 1', 'b = 1'),
            (None, 'return 2', 'b = 2'),
            (('return 2', 'b = 1'), 'b = 2', None),
            (None, 'return b', None),
        ),
    )
    self.assertGraphEnds(
        graph, 'a', ('return b', 'b = 2'))

  def test_raise_adds_finally_sortcuts(self):
    # [FUNCTION DOC]
    # (The name contains a typo: "sortcuts" for "shortcuts".)
    # Scenario: a conditional raise in nested try/finally. Expects the inner finally block to have
    # edges both to the code after it and directly to the outer finally (the shortcut taken when
    # an exception is raised). Exits are the final return and the outer finally block.

    def test_fn(a):
      try:
        try:
          if a > 0:
            raise a
          c = 1
        finally:
          b = 1
        c = 2
      finally:
        b = 2
      return b, c

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', '(a > 0)', ('raise a', 'c = 1')),
            ('(a > 0)', 'raise a', 'b = 1'),
            ('(a > 0)', 'c = 1', 'b = 1'),
            (('raise a', 'c = 1'), 'b = 1', ('c = 2', 'b = 2')),
            ('b = 1', 'c = 2', 'b = 2'),
            (('b = 1', 'c = 2'), 'b = 2', 'return (b, c)'),
            ('b = 2', 'return (b, c)', None),
        ),
    )
    self.assertGraphEnds(
        graph, 'a', ('return (b, c)', 'b = 2'))

  def test_raise_exits_via_except(self):
    # [FUNCTION DOC]
    # Scenario: a raise in a try with two handlers and a finally block. Expects the raise to lead
    # to each handler and directly to the finally block, with the finally block as the sole exit.

    def test_fn(a, b):
      try:
        raise b
      except a:
        c = 1
      except b:
        c = 2
      finally:
        c += 3

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a, b', 'raise b', ('c = 1', 'c = 2', 'c += 3')),
            ('raise b', 'c = 1', 'c += 3'),
            ('raise b', 'c = 2', 'c += 3'),
            (('raise b', 'c = 1', 'c = 2'), 'c += 3', None),
        ),
    )
    self.assertGraphEnds(graph, 'a, b', ('c += 3',))

  def test_list_comprehension(self):
    # [FUNCTION DOC]
    # Scenario: an assignment from a list comprehension, then a return. Expects the comprehension
    # to be part of its statement node (no separate graph), giving a simple chain.

    def test_fn(a):
      c = [b for b in a]
      return c

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a', 'c = [b for b in a]', 'return c'),
            ('c = [b for b in a]', 'return c', None),
        ),
    )
    self.assertGraphEnds(graph, 'a', ('return c',))

  def test_class_definition_empty(self):
    # [FUNCTION DOC]
    # Scenario: a class definition with an empty body inside a function, followed by a return.
    # Expects the class definition to be a single node in the chain.

    def test_fn(a, b):
      class C(a(b)):
        pass
      return C

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a, b', 'class C', 'return C'),
            ('class C', 'return C', None),
        ),
    )
    self.assertGraphEnds(graph, 'a, b', ('return C',))

  def test_class_definition_with_members(self):
    # [FUNCTION DOC]
    # Scenario: a class definition with a member assignment inside a function. Expects the same
    # single class node; the members are not separate graph nodes.

    def test_fn(a, b):
      class C(a(b)):
        d = 1
      return C

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('a, b', 'class C', 'return C'),
            ('class C', 'return C', None),
        ),
    )
    self.assertGraphEnds(graph, 'a, b', ('return C',))

  def test_import(self):
    # [FUNCTION DOC]
    # Scenario: a function with no arguments containing an import statement then a return.
    # Expects the entry node to print as empty and a chain entry, import, return.

    def test_fn():
      from a import b  # pylint:disable=g-import-not-at-top
      return b

    graph, = self._build_cfg(test_fn).values()

    self.assertGraphMatches(
        graph,
        (
            ('', 'from a import b', 'return b'),
            ('from a import b', 'return b', None),
        ),
    )
    self.assertGraphEnds(graph, '', ('return b',))


if __name__ == '__main__':
  test.main()
