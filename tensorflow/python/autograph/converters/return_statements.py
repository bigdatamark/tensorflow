# [MODULE DOC]
# AutoGraph converter that canonicalizes functions so they have a single return statement at the
# end. Return statements are lowered into assignments to a "do return" flag variable and a return
# value variable, later statements are guarded by conditionals on the flag, and loop conditions
# are extended to stop once a return happened. Exposes a transform entry function plus two AST
# transformer classes (a conditional-return rewriter and the return lowering transformer) and
# annotation key constants. Requires the functions converter to have run first when missing
# returns are allowed. No import-time side effects beyond defining constants.
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
"""Canonicalizes functions with multiple returns to use just one."""

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno


BODY_DEFINITELY_RETURNS = 'BODY_DEFINITELY_RETURNS'
ORELSE_DEFINITELY_RETURNS = 'ORELSE_DEFINITELY_RETURNS'
STMT_DEFINITELY_RETURNS = 'STMT_DEFINITELY_RETURNS'


class _RewriteBlock(object):
  # [CLASS DOC]
  # HELPER: per-block state holder for the conditional return rewriter.
  # Scope-stack state object tracking whether the block currently being visited is known to
  # definitely return on every path. Single attribute: definitely_returns, initially false.

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: initializer for the block state.
    # Sets the definitely-returns flag to false. Takes no arguments, returns nothing.
    self.definitely_returns = False


class ConditionalReturnRewriter(converter.Base):
  # [CLASS DOC]
  # First-pass AST transformer run by the transform function. It detects conditional statements
  # whose body (or else branch) definitely returns, and moves the statements that follow the
  # conditional into the opposite branch, so that all paths visibly return and fewer null return
  # values are needed. It records its findings as node annotations (body/else/statement
  # definitely returns) and tracks nested block state with a block state object. Subclass of
  # the converter base transformer; created with the transformation context.
  """Rewrites a pattern where it's unobvious that all paths return a value.

  This rewrite allows avoiding intermediate None return values.

  The following pattern:

      if cond:
        <block 1>
        return
      else:
        <block 2>
      <block 3>

  is converted to:

      if cond:
        <block 1>
        return
      else:
        <block 2>
        <block 3>

  and vice-versa (if the else returns, subsequent statements are moved under the
  if branch).
  """

  def visit_Return(self, node):
    # [FUNCTION DOC]
    # Visitor for return statements: marks the current block as definitely returning and returns
    # the node unchanged.
    self.state[_RewriteBlock].definitely_returns = True
    return node

  def _postprocess_statement(self, node):
    # [FUNCTION DOC]
    # HELPER: after-visit callback used when visiting statement blocks in the rewriter.
    # If the statement is annotated as definitely returning, marks the current block likewise.
    # If the statement is a conditional whose body definitely returns, returns the statement
    # together with its else statements as the nodes to be moved after it (so the following
    # statements are appended to the else branch); if instead its else branch definitely
    # returns, pairs it with its body statements. Otherwise returns the node and nothing.
    # Input: a statement node. Output: pair of (node, block to extend or nothing).
    # If the node definitely returns (e.g. it's a with statement with a
    # return statement in it), then the current block also definitely returns.
    if anno.getanno(node, STMT_DEFINITELY_RETURNS, default=False):
      self.state[_RewriteBlock].definitely_returns = True

    # The special case: collapse a typical conditional return pattern into
    # a single conditional with possibly returns on both branches. This
    # reduces the use of None return values, which don't work with TF
    # conditionals.
    if (isinstance(node, gast.If)
        and anno.getanno(node, BODY_DEFINITELY_RETURNS, default=False)):
      return node, node.orelse
    elif (isinstance(node, gast.If)
          and anno.getanno(node, ORELSE_DEFINITELY_RETURNS, default=False)):
      return node, node.body

    return node, None

  def _visit_statement_block(self, node, nodes):
    # [FUNCTION DOC]
    # HELPER: visits a list of statements as a new nested block for the rewriter.
    # Enters a new block state, visits the statements with the post-processing callback, reads
    # whether the block definitely returns, and leaves the block. Inputs: the owning node
    # (unused) and the list of statements. Returns a pair: the new statement list and a boolean
    # telling whether the block definitely returns.
    self.state[_RewriteBlock].enter()
    new_nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)
    block_definitely_returns = self.state[_RewriteBlock].definitely_returns
    self.state[_RewriteBlock].exit()
    return new_nodes, block_definitely_returns

  def visit_While(self, node):
    # [FUNCTION DOC]
    # Visitor for while loops: visits the test, then rewrites the body and else blocks as nested
    # blocks, ignoring whether they definitely return. Returns the modified node.
    node.test = self.visit(node.test)
    node.body, _ = self._visit_statement_block(node, node.body)
    node.orelse, _ = self._visit_statement_block(node, node.orelse)
    return node

  def visit_For(self, node):
    # [FUNCTION DOC]
    # Visitor for for loops: visits the iterable and target, then rewrites the body and else
    # blocks as nested blocks, ignoring whether they definitely return. Returns the node.
    node.iter = self.visit(node.iter)
    node.target = self.visit(node.target)
    node.body, _ = self._visit_statement_block(node, node.body)
    node.orelse, _ = self._visit_statement_block(node, node.orelse)
    return node

  def visit_With(self, node):
    # [FUNCTION DOC]
    # Visitor for with statements: visits the context items, rewrites the body as a nested
    # block, and if the body definitely returns annotates the statement as definitely returning
    # so the enclosing block is also marked. Returns the node.
    node.items = self.visit_block(node.items)
    node.body, definitely_returns = self._visit_statement_block(node, node.body)
    if definitely_returns:
      anno.setanno(node, STMT_DEFINITELY_RETURNS, True)
    return node

  def visit_Try(self, node):
    # [FUNCTION DOC]
    # Visitor for try statements: rewrites the body, else and finally blocks as nested blocks and
    # visits the exception handlers. Does not annotate the statement as definitely returning,
    # since a try block may raise. Returns the node.
    # We could decide whether a 'try' DEFINITELY_RETURNS based on its components
    # It is not clear whether we want to do anything with this given
    # a 'try' is likely to throw an exception in some circumstances.
    node.body, _ = self._visit_statement_block(node, node.body)
    node.orelse, _ = self._visit_statement_block(node, node.orelse)
    node.finalbody, _ = self._visit_statement_block(node, node.finalbody)
    node.handlers = self.visit_block(node.handlers)
    return node

  def visit_ExceptHandler(self, node):
    # [FUNCTION DOC]
    # Visitor for exception handlers: rewrites the handler body as a nested block, ignoring
    # whether it returns. Returns the node.
    # To determine whether `try` DEFINITELY_RETURNS we need to revisit this.
    node.body, _ = self._visit_statement_block(node, node.body)
    return node

  def visit_If(self, node):
    # [FUNCTION DOC]
    # Visitor for conditionals: visits the test, rewrites the body and else blocks, and
    # annotates the node when its body and/or its else branch definitely return. If both
    # branches definitely return, the enclosing block is marked as definitely returning.
    # The annotations drive the statement post-processing that moves following statements into
    # the other branch. Returns the node.
    node.test = self.visit(node.test)

    node.body, body_definitely_returns = self._visit_statement_block(
        node, node.body)
    if body_definitely_returns:
      anno.setanno(node, BODY_DEFINITELY_RETURNS, True)

    node.orelse, orelse_definitely_returns = self._visit_statement_block(
        node, node.orelse)
    if orelse_definitely_returns:
      anno.setanno(node, ORELSE_DEFINITELY_RETURNS, True)

    if body_definitely_returns and orelse_definitely_returns:
      self.state[_RewriteBlock].definitely_returns = True

    return node

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Visitor for function definitions: visits the arguments and rewrites the function body as a
    # nested block. Returns the node.
    node.args = self.visit(node.args)
    node.body, _ = self._visit_statement_block(node, node.body)
    return node


class _Block(object):
  # [CLASS DOC]
  # HELPER: per-block state for the return statements transformer.
  # State object pushed on a scope stack for each statement block. Attributes: whether the block
  # is a function body, whether a return was used inside it, and two flags controlling when the
  # next statement must be wrapped in a guard conditional (create guard next / now).

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: initializer for the block state.
    # Sets the function-body, return-used, create-guard-next and create-guard-now flags all to
    # false. Takes no arguments, returns nothing.
    self.is_function = False
    self.return_used = False
    self.create_guard_next = False
    self.create_guard_now = False

  def __repr__(self):
    # [FUNCTION DOC]
    # HELPER: debugging representation. Returns a string saying whether a return was used.
    return 'used: {}'.format(
        self.return_used)


class _Function(object):
  # [CLASS DOC]
  # HELPER: per-function state for the return statements transformer.
  # Holds the generated variable names used for the "do return" control flag and the return
  # value of the function currently being transformed. Both start unset.

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: initializer; sets both generated variable names to nothing. Returns nothing.
    self.do_return_var_name = None
    self.retval_var_name = None

  def __repr__(self):
    # [FUNCTION DOC]
    # HELPER: debugging representation. Returns a string with the control and value names.
    return 'return control: {}, return value: {}'.format(
        self.do_return_var_name, self.retval_var_name)


class ReturnStatementsTransformer(converter.Base):
  # [CLASS DOC]
  # Second-pass AST transformer that rewrites every return statement into assignments of a
  # generated control flag and return value variable (wrapped so the flag resets if evaluating
  # the value raises), guards following statements with a check on the flag, adds the flag to
  # while-loop conditions and to for-loop extra tests, and appends a single final return per
  # function. Constructed with the transformation context and a flag allow_missing_return which
  # selects whether functions that may fall off the end return a special undefined value through
  # the function context (requires the functions converter's wrapping with statement) or just
  # the value variable. Uses block and function state objects on its state stack.
  """Lowers return statements into variables and conditionals.

  Specifically, the following pattern:

      <block 1>
      return val
      <block 2>

  is converted to:

      do_return = False
      retval = None

      <block 1>

      do_return = True
      retval = val

      if not do_return:
        <block 2>

      return retval

  The conversion adjusts loops as well:

      <block 1>
      while cond:
        <block 2>
        return retval

  is converted to:

      <block 1>
      while not do_return and cond:
        <block 2>
        do_return = True
        retval = val
  """

  def __init__(self, ctx, allow_missing_return):
    # [FUNCTION DOC]
    # Constructor. Inputs: the transformation context passed to the base class, and a boolean
    # allow_missing_return deciding how function endings are built. Stores the boolean.
    super(ReturnStatementsTransformer, self).__init__(ctx)
    self.allow_missing_return = allow_missing_return

  def visit_Return(self, node):
    # [FUNCTION DOC]
    # Visitor for return statements. Marks every enclosing block up to and including the function
    # body as having used a return and as needing a guard around the next statement. Replaces
    # the statement with a try construct that sets the control flag true and stores the returned
    # value (nothing becomes a null value); if evaluating the value raises, the flag is reset to
    # false and the error is re-raised. Returns the new statement list.
    for block in reversed(self.state[_Block].stack):
      block.return_used = True
      block.create_guard_next = True
      if block.is_function:
        break

    retval = node.value if node.value else parser.parse_expression('None')

    # Note: If `return <expr> raises, then the return is aborted.
    # The try-catch below ensures the variables remain consistent in that case.
    template = """
      try:
        do_return_var_name = True
        retval_var_name = retval
      except:
        do_return_var_name = False
        raise
    """
    node = templates.replace(
        template,
        do_return_var_name=self.state[_Function].do_return_var_name,
        retval_var_name=self.state[_Function].retval_var_name,
        retval=retval)

    return node

  def _postprocess_statement(self, node):
    # [FUNCTION DOC]
    # HELPER: after-visit callback used when visiting statement blocks in this transformer.
    # If no return has been used in the block yet, returns the node unchanged. Otherwise, if a
    # guard is due for this statement, wraps it in a conditional that runs only while the control
    # flag is false and returns that conditional with its body as the block to continue
    # appending following statements to. Then shifts the pending guard flag (next becomes now,
    # next is cleared). Input: statement node. Output: pair of (node, block or nothing).
    if not self.state[_Block].return_used:
      return node, None

    state = self.state[_Block]
    if state.create_guard_now:
      template = """
        if not do_return_var_name:
          original_node
      """
      cond, = templates.replace(
          template,
          do_return_var_name=self.state[_Function].do_return_var_name,
          original_node=node)
      node, block = cond, cond.body
    else:
      node, block = node, None

    state.create_guard_now = state.create_guard_next
    state.create_guard_next = False

    return node, block

  def _visit_statement_block(self, node, nodes):
    # [FUNCTION DOC]
    # HELPER: visits a list of statements as a new nested block with guard post-processing.
    # Enters a block state, visits the statements, leaves the block, and returns the new
    # statement list. The owning node argument is unused.
    self.state[_Block].enter()
    nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)
    self.state[_Block].exit()
    return nodes

  def visit_While(self, node):
    # [FUNCTION DOC]
    # Visitor for while loops. Visits the test, rewrites the body, and if a return was used in
    # the enclosing block, changes the loop condition to also require that the control flag is
    # not set. Rewrites the else block and returns the node.
    node.test = self.visit(node.test)

    # Add the check for return to the loop condition.
    node.body = self._visit_statement_block(node, node.body)
    if self.state[_Block].return_used:
      node.test = templates.replace_as_expression(
          'not control_var and test',
          test=node.test,
          control_var=self.state[_Function].do_return_var_name)

    node.orelse = self._visit_statement_block(node, node.orelse)
    return node

  def visit_For(self, node):
    # [FUNCTION DOC]
    # Visitor for for loops. Visits iterable and target, rewrites the body, and if a return was
    # used, sets (or extends with a logical and) the loop's extra-test annotation so that
    # iteration stops once the control flag is set. Rewrites the else block and returns the node.
    node.iter = self.visit(node.iter)
    node.target = self.visit(node.target)

    # Add the check for return to the loop condition.
    node.body = self._visit_statement_block(node, node.body)
    if self.state[_Block].return_used:
      extra_test = anno.getanno(node, anno.Basic.EXTRA_LOOP_TEST, default=None)
      if extra_test is not None:
        extra_test = templates.replace_as_expression(
            'not control_var and extra_test',
            extra_test=extra_test,
            control_var=self.state[_Function].do_return_var_name)
      else:
        extra_test = templates.replace_as_expression(
            'not control_var',
            control_var=self.state[_Function].do_return_var_name)
      anno.setanno(node, anno.Basic.EXTRA_LOOP_TEST, extra_test)

    node.orelse = self._visit_statement_block(node, node.orelse)
    return node

  def visit_With(self, node):
    # [FUNCTION DOC]
    # Visitor for with statements: visits the context items and rewrites the body as a nested
    # block. Returns the node.
    node.items = self.visit_block(node.items)
    node.body = self._visit_statement_block(node, node.body)
    return node

  def visit_Try(self, node):
    # [FUNCTION DOC]
    # Visitor for try statements: rewrites body, else and finally blocks as nested blocks and
    # visits the handlers. Returns the node.
    node.body = self._visit_statement_block(node, node.body)
    node.orelse = self._visit_statement_block(node, node.orelse)
    node.finalbody = self._visit_statement_block(node, node.finalbody)
    node.handlers = self.visit_block(node.handlers)
    return node

  def visit_ExceptHandler(self, node):
    # [FUNCTION DOC]
    # Visitor for exception handlers: rewrites the handler body as a nested block. Returns node.
    node.body = self._visit_statement_block(node, node.body)
    return node

  def visit_If(self, node):
    # [FUNCTION DOC]
    # Visitor for conditionals: visits the test and rewrites body and else as nested blocks.
    # Returns the node.
    node.test = self.visit(node.test)
    node.body = self._visit_statement_block(node, node.body)
    node.orelse = self._visit_statement_block(node, node.orelse)
    return node

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Visitor for function definitions: the main driver of the return lowering. Opens function and
    # block state, generates unique names for the control flag ("do_return") and the value
    # variable ("retval_") that avoid names referenced in the body, and rewrites the body. If a
    # return was used: with allow_missing_return, the last body statement must be the wrapping
    # with statement from the functions converter (an assertion fails otherwise); its body is
    # replaced by flag initialization to false, value initialization to an undefined-return
    # marker, the original body, and a final return through the function context's return helper.
    # Without allow_missing_return, the body is followed by a plain return of the value variable.
    # Returns the node.
    with self.state[_Function] as fn:
      with self.state[_Block] as block:
        block.is_function = True

        scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
        do_return_var_name = self.ctx.namer.new_symbol('do_return',
                                                       scope.referenced)
        retval_var_name = self.ctx.namer.new_symbol('retval_', scope.referenced)
        fn.do_return_var_name = do_return_var_name
        fn.retval_var_name = retval_var_name

        node.body = self._visit_statement_block(node, node.body)

        if block.return_used:

          if self.allow_missing_return:
            # The function would have a single `with` node that wraps the
            # entire body. If the function had a docstring, the body has two
            # nodes, with the `with` as the second node.
            wrapper_node = node.body[-1]
            assert isinstance(wrapper_node, gast.With), (
                'This transformer requires the functions converter.')

            template = """
              do_return_var_name = False
              retval_var_name = ag__.UndefinedReturnValue()
              body
              return function_context.ret(retval_var_name, do_return_var_name)
            """

            wrapper_node.body = templates.replace(
                template,
                body=wrapper_node.body,
                do_return_var_name=do_return_var_name,
                function_context=anno.getanno(node, 'function_context_name'),
                retval_var_name=retval_var_name)
          else:
            template = """
              body
              return retval_var_name
            """
            node.body = templates.replace(
                template,
                body=node.body,
                do_return_var_name=do_return_var_name,
                retval_var_name=retval_var_name)

    return node


def transform(node, ctx, default_to_null_return=True):
  # [FUNCTION DOC]
  # Entry point of the converter. Resolves qualified names and activity analysis on the tree, runs
  # the conditional return rewriter, re-resolves names and activity (the tree changed), then runs
  # the return lowering transformer with allow_missing_return set from default_to_null_return
  # (a boolean, default true). Inputs: the AST node of the function(s), the transformation
  # context. Returns the transformed AST node.
  """Ensure a function has only a single return, at the end."""
  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx, None)

  # Note: Technically, these two could be merged into a single walk, but
  # keeping them separate helps with readability.
  node = ConditionalReturnRewriter(ctx).visit(node)

  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx, None)
  transformer = ReturnStatementsTransformer(
      ctx, allow_missing_return=default_to_null_return)
  node = transformer.visit(node)
  return node
