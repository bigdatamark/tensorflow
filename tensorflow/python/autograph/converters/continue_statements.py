# [MODULE DOC]
# AutoGraph converter that removes continue statements by replacing each with the setting of a
# generated boolean control variable and wrapping subsequent statements in enclosing blocks in
# conditionals on that variable. Exposes ContinueCanonicalizationTransformer and transform; two
# private helper classes hold per-loop and per-block bookkeeping.
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
"""Canonicalizes continue statements by de-sugaring into a control boolean."""

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno


class _Continue(object):
  # [CLASS DOC]
  # HELPER: per-loop state record for ContinueCanonicalizationTransformer. Holds whether a
  # continue was seen in the loop and the generated control variable name.

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the record: continue not used, control variable name unset (null).
    self.used = False
    self.control_var_name = None

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a debugging text showing the used flag and control variable name.
    return '<_Continue(used: {}, var: {})>'.format(self.used,
                                                   self.control_var_name)


class _Block(object):
  # [CLASS DOC]
  # HELPER: per-block bookkeeping record for ContinueCanonicalizationTransformer, kept on a stack
  # of lexical blocks. Flags: is_loop_type (block is a loop body), create_guard_current (wrap the
  # statement being processed in a guard conditional) and create_guard_next (wrap the following
  # statement).
  """Tracks information about lexical blocks as they are visited in the AST.

  Mainly, this object tracks the creation of block guards that replace
  `continue` statements (e.g. `if not continue_:`).

  Attributes:
    create_guard_current: bool, whether to create a guard for the current
      statement.
    create_guard_next: bool, whether to create a guard for the next
      statement.
    is_loop_type: bool, whether this block is the body of a loop.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the block record: not a loop body, and no guard to be created for the current
    # or the next statement.
    self.is_loop_type = False
    self.create_guard_current = False
    self.create_guard_next = False


class ContinueCanonicalizationTransformer(converter.Base):
  # [CLASS DOC]
  # Syntax-tree transformer (subclass of the converter base) that rewrites continue statements
  # into assignments to a generated control variable, initialized false at the top of each loop
  # body, with guard conditionals around later statements in affected blocks. Uses state stacks
  # of continue records and block records. Created with a conversion context by transform.
  """Canonicalizes continue statements into additional conditionals."""

  def visit_Continue(self, node):
    # [FUNCTION DOC]
    # Visitor for a continue node. Marks the current loop record as using continue, and flags
    # every enclosing block up to and including the innermost loop body so the next statement in
    # each gets guarded. Returns a replacement statement setting the control variable to true.
    self.state[_Continue].used = True
    for block in reversed(self.state[_Block].stack):
      # See ContinueCanonicalizationTest.test_multiple_continues for an example
      # it's necessary to create guards for all enclosing affected blocks, not
      # just that of the current block.
      block.create_guard_next = True
      if block.is_loop_type:
        # continue only affects the innermost loop
        break
    template = """
      var_name = True
    """
    return templates.replace(
        template, var_name=self.state[_Continue].control_var_name)

  def _postprocess_statement(self, node):
    # [FUNCTION DOC]
    # HELPER: callback run after each statement of a block is visited (by the block visitor).
    # Input: the visited statement. If a continue has been used, shifts the pending guard flags
    # (next becomes current), and if the statement needed a guard wraps it in a conditional that
    # runs it only when the control variable is false. Returns a pair: the (possibly wrapped)
    # statement and the guarded body list (to continue adding following statements into), or null
    # when not wrapped.
    if self.state[_Continue].used:
      block = self.state[_Block]
      should_wrap_current = block.create_guard_current
      # After processing propagate whether to guard the next statement
      block.create_guard_current = block.create_guard_next
      block.create_guard_next = False
      if should_wrap_current:
        template = """
          if not var_name:
            original_node
        """
        cond, = templates.replace(
            template,
            var_name=self.state[_Continue].control_var_name,
            original_node=node)
        return cond, cond.body
    return node, None

  def _visit_loop_body(self, node, nodes):
    # [FUNCTION DOC]
    # HELPER: transforms the body of a while or for loop.
    # Inputs: node (the loop) and nodes (its body statements). Pushes continue and block records
    # (block marked as loop), creates a fresh control variable name avoiding names referenced in
    # the loop body, visits the statements with the post-processing callback, and if a continue
    # was used prepends an initialization of the variable to false. Pops the records. Returns the
    # new list of body statements.
    self.state[_Continue].enter()
    self.state[_Block].enter()
    self.state[_Block].is_loop_type = True
    scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    continue_var = self.ctx.namer.new_symbol('continue_', scope.referenced)
    self.state[_Continue].control_var_name = continue_var

    nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)

    if self.state[_Continue].used:
      template = """
        var_name = False
      """
      control_var_init = templates.replace(template, var_name=continue_var)
      nodes = control_var_init + nodes

    self.state[_Block].exit()
    self.state[_Continue].exit()
    return nodes

  def _visit_non_loop_body(self, nodes):
    # [FUNCTION DOC]
    # HELPER: transforms a non-loop statement list (conditional, with, try or else bodies).
    # Input: list of statements. Pushes a block record, visits the statements with the
    # post-processing callback, pops it. Returns the new list.
    self.state[_Block].enter()
    nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)
    self.state[_Block].exit()
    return nodes

  def visit_While(self, node):
    # [FUNCTION DOC]
    # Visitor for while loops: visits the test, transforms the body as a loop body, and the else
    # clause as a non-loop body (a continue there applies to the enclosing loop). Returns the node.
    node.test = self.visit(node.test)
    node.body = self._visit_loop_body(node, node.body)
    # A continue in the else clause applies to the containing scope.
    node.orelse = self._visit_non_loop_body(node.orelse)
    return node

  def visit_For(self, node):
    # [FUNCTION DOC]
    # Visitor for for loops: visits target and iterable, transforms the body as a loop body and
    # the else clause as a non-loop body. Returns the node.
    node.target = self.generic_visit(node.target)
    node.iter = self.generic_visit(node.iter)
    node.body = self._visit_loop_body(node, node.body)
    # A continue in the else clause applies to the containing scope.
    node.orelse = self._visit_non_loop_body(node.orelse)
    return node

  def visit_If(self, node):
    # [FUNCTION DOC]
    # Visitor for conditionals: transforms the body and else clause as non-loop bodies. The test
    # expression is not visited. Returns the node.
    node.body = self._visit_non_loop_body(node.body)
    node.orelse = self._visit_non_loop_body(node.orelse)
    return node

  def visit_With(self, node):
    # [FUNCTION DOC]
    # Visitor for with statements: visits the context items and transforms the body as a non-loop
    # body. Returns the node.
    node.items = self.visit_block(node.items)
    node.body = self._visit_non_loop_body(node.body)
    return node

  def visit_Try(self, node):
    # [FUNCTION DOC]
    # Visitor for try statements: transforms body, else clause and final clause as non-loop bodies
    # (continue is allowed in final clauses in newer language versions) and visits the exception
    # handlers. Returns the node.
    node.body = self._visit_non_loop_body(node.body)
    node.orelse = self._visit_non_loop_body(node.orelse)
    # In Python 3.8 and later continue is allowed in finally blocks
    node.finalbody = self._visit_non_loop_body(node.finalbody)
    node.handlers = self.visit_block(node.handlers)
    return node

  def visit_ExceptHandler(self, node):
    # [FUNCTION DOC]
    # Visitor for exception handlers: transforms the handler body as a non-loop body. Returns the
    # node.
    node.body = self._visit_non_loop_body(node.body)
    return node


def transform(node, ctx):
  # [FUNCTION DOC]
  # Converter entry point. Inputs: node (syntax tree) and ctx (conversion context). Resolves
  # qualified names and variable activity analysis, then applies the continue transformer.
  # Returns the converted tree.
  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx, None)

  node = ContinueCanonicalizationTransformer(ctx).visit(node)
  return node
