# [MODULE DOC]
# AutoGraph converter that lowers break statements in while and for loops into a boolean control
# variable, extra loop-test conditions and guarded conditionals, so loops can be staged into
# graph code. Exposes the BreakTransformer and the transform entry point; a private state class
# tracks per-loop break usage.
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
"""Lowers break statements to conditionals."""

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno


class _Break(object):
  # [CLASS DOC]
  # HELPER: per-loop state record for BreakTransformer, stored in its state stack.
  # Holds a flag telling whether a break was seen in the loop body and the name of the generated
  # control variable that replaces it.

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the record: break not used, control variable name unset (null).
    self.used = False
    self.control_var_name = None

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a debugging text showing whether a break was used and the control variable name.
    return 'used: %s, var: %s' % (self.used, self.control_var_name)


class BreakTransformer(converter.Base):
  # [CLASS DOC]
  # Syntax-tree transformer (subclass of the converter base) that rewrites break statements: each
  # break becomes an assignment setting a generated boolean variable followed by a continue; the
  # loop test (while) or an extra-loop-test annotation (for) checks the variable, and trailing
  # else clauses are guarded so they run only when no break happened. Uses a stack of break
  # state records in the inherited state storage. Created with a conversion context.
  """Canonicalizes break statements into additional conditionals."""

  def visit_Break(self, node):
    # [FUNCTION DOC]
    # Visitor for a break node. Marks the current loop state as having a break and returns
    # replacement statements: set the loop's control variable to true, then continue.
    # KNOWN ISSUE: a source comment notes this fails when expanded inside a top-level else block
    # of the loop.
    self.state[_Break].used = True
    var_name = self.state[_Break].control_var_name
    # TODO(mdan): This will fail when expanded inside a top-level else block.
    template = """
      var_name = True
      continue
    """
    return templates.replace(template, var_name=var_name)

  def _guard_if_present(self, block, var_name):
    # [FUNCTION DOC]
    # HELPER: wraps loop else-clauses for visit_While and visit_For.
    # Inputs: block (list of statements, possibly empty) and var_name (name of control variable).
    # Returns the block unchanged if empty, otherwise a conditional that runs the block only when
    # the variable is false.
    """Prevents the block from executing if var_name is set."""
    if not block:
      return block

    template = """
        if not var_name:
          block
      """
    node = templates.replace(
        template,
        var_name=var_name,
        block=block)
    return node

  def _process_body(self, nodes, break_var):
    # [FUNCTION DOC]
    # HELPER: transforms a loop body for visit_While and visit_For.
    # Inputs: nodes (list of body statements) and break_var (generated control variable name).
    # Pushes a new break state, records the variable name, visits the block, reads whether a
    # break was used, pops the state. Returns a pair of the transformed nodes and the used flag.
    self.state[_Break].enter()
    self.state[_Break].control_var_name = break_var
    nodes = self.visit_block(nodes)
    break_used = self.state[_Break].used
    self.state[_Break].exit()
    return nodes, break_used

  def visit_While(self, node):
    # [FUNCTION DOC]
    # Visitor for while loops. Creates a fresh control variable name, transforms test, body and
    # else clause (a break in the else belongs to the enclosing scope). If no break was used,
    # rebuilds the plain loop. Otherwise emits: set variable false, loop with test requiring the
    # variable false and the original test, and the else clause guarded by the variable. Copies
    # directive annotations from the original to the new loop node. Returns the replacement nodes.
    original_node = node
    scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    break_var = self.ctx.namer.new_symbol('break_', scope.referenced)

    node.test = self.visit(node.test)
    node.body, break_used = self._process_body(node.body, break_var)
    # A break in the else clause applies to the containing scope.
    node.orelse = self.visit_block(node.orelse)

    if not break_used:
      template = """
        while test:
          body
        orelse
      """
      node = templates.replace(
          template, test=node.test, body=node.body, orelse=node.orelse)

      new_while_node = node[0]
      anno.copyanno(original_node, new_while_node, anno.Basic.DIRECTIVES)

      return node

    # Python's else clause only triggers if the loop exited cleanly (e.g.
    # break did not trigger).
    guarded_orelse = self._guard_if_present(node.orelse, break_var)

    template = """
      var_name = False
      while not var_name and test:
        body
      orelse
    """
    node = templates.replace(
        template,
        var_name=break_var,
        test=node.test,
        body=node.body,
        orelse=guarded_orelse)

    new_while_node = node[1]
    anno.copyanno(original_node, new_while_node, anno.Basic.DIRECTIVES)

    return node

  def visit_For(self, node):
    # [FUNCTION DOC]
    # Visitor for for loops. Creates a control variable name, transforms target, iterable, body and
    # else clause. If no break was used, rebuilds the loop, copying the extra-loop-test and
    # directive annotations. Otherwise emits: set variable false, a loop whose body starts with a
    # no-op use of the variable (so static analysis sees it), annotates the new loop with an
    # extra test requiring the variable false, copies directives, and guards the else clause.
    # Returns the replacement nodes.
    original_node = node
    scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    break_var = self.ctx.namer.new_symbol('break_', scope.referenced)

    node.target = self.visit(node.target)
    node.iter = self.visit(node.iter)
    node.body, break_used = self._process_body(node.body, break_var)
    # A break in the else clause applies to the containing scope.
    node.orelse = self.visit_block(node.orelse)

    if not break_used:
      template = """
        for target in iter_:
          body
        orelse
      """
      node = templates.replace(
          template,
          iter_=node.iter,
          target=node.target,
          body=node.body,
          orelse=node.orelse)

      new_for_node = node[0]
      anno.copyanno(original_node, new_for_node, anno.Basic.EXTRA_LOOP_TEST)
      anno.copyanno(original_node, new_for_node, anno.Basic.DIRECTIVES)

      return node

    # Python's else clause only triggers if the loop exited cleanly (e.g.
    # break did not trigger).
    guarded_orelse = self._guard_if_present(node.orelse, break_var)
    extra_test = templates.replace_as_expression(
        'not var_name', var_name=break_var)

    # The extra test is hidden in the AST, which will confuse the static
    # analysis. To mitigate that, we insert a no-op statement that ensures
    # the control variable is marked as used.
    # TODO(mdan): Use a marker instead, e.g. ag__.condition_loop_on(var_name)
    template = """
      var_name = False
      for target in iter_:
        (var_name,)
        body
      orelse
    """
    node = templates.replace(
        template,
        var_name=break_var,
        iter_=node.iter,
        target=node.target,
        body=node.body,
        orelse=guarded_orelse)

    new_for_node = node[1]
    anno.setanno(new_for_node, anno.Basic.EXTRA_LOOP_TEST, extra_test)
    anno.copyanno(original_node, new_for_node, anno.Basic.DIRECTIVES)

    return node


def transform(node, ctx):
  # [FUNCTION DOC]
  # Converter entry point. Inputs: node (syntax tree) and ctx (conversion context). Resolves
  # qualified names and variable activity analysis on the tree, then applies the break
  # transformer. Returns the converted tree.
  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx, None)

  transformer = BreakTransformer(ctx)
  node = transformer.visit(node)
  return node
