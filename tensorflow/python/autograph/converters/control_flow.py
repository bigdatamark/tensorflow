# [MODULE DOC]
# Autograph converter that rewrites Python if, while and for statements into calls to the
# autograph runtime operators (if statement, while statement, for statement) so they can become
# TensorFlow control flow. It generates nested body/test functions plus state getter/setter
# functions that expose the variables a block modifies, using static analysis results (liveness,
# reaching definitions, activity) to decide which variables are block state. Exposes the
# transform entry function. No import-time side effects.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Handles control flow statements: while, for, if."""

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
from tensorflow.python.autograph.pyct.static_analysis import liveness
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs


class _Function(object):
  # [CLASS DOC]
  # HELPER: per-function state record for ControlFlowTransformer. Holds a single class attribute,
  # scope, which is set to the activity scope of the function or lambda currently being visited
  # (nothing by default). Used via the transformer state stack, not instantiated directly.

  scope = None


class ControlFlowTransformer(converter.Base):
  # [CLASS DOC]
  # AST transformer converting if, while and for nodes into autograph runtime calls with generated
  # helper functions. Subclass of the autograph converter base class; it tracks the enclosing
  # function scope through a state stack of _Function records and uses the context namer to
  # create unique symbol names. Created by the transform function and applied by visiting a tree.
  """Transforms control flow structures like loops an conditionals."""

  def visit_Lambda(self, node):
    # [FUNCTION DOC]
    # Visits a lambda node: pushes a function state whose scope is the lambda's scope annotation,
    # then visits children and returns the (possibly transformed) node.
    with self.state[_Function] as fn:
      fn.scope = anno.getanno(node, anno.Static.SCOPE)
      return self.generic_visit(node)

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Visits a function definition: pushes a function state whose scope is the body scope
    # annotation of the node, then visits children and returns the transformed node.
    with self.state[_Function] as fn:
      fn.scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
      return self.generic_visit(node)

  def _create_nonlocal_declarations(self, vars_):
    # [FUNCTION DOC]
    # HELPER: builds the scope declaration statements for generated nested functions.
    # Takes a collection of qualified-name variables. Returns a list with at most two statements:
    # a global declaration for variables that are globals of the enclosing function, and a
    # nonlocal declaration for the remaining non-composite variables. Composite names (such as
    # attribute accesses) are skipped. Returns an empty list if there is nothing to declare.
    vars_ = set(vars_)
    results = []
    global_vars = self.state[_Function].scope.globals & vars_

    if global_vars:
      results.append(gast.Global([str(v) for v in global_vars]))

    nonlocal_vars = [
        v for v in vars_ if not v.is_composite() and v not in global_vars]
    if nonlocal_vars:
      results.append(gast.Nonlocal([str(v) for v in nonlocal_vars]))

    return results

  def _create_state_functions(
      self, block_vars, nonlocal_declarations, getter_name, setter_name):
    # [FUNCTION DOC]
    # HELPER: generates the state getter and setter function definitions for a control flow block.
    # Inputs: block_vars, the ordered block state variables; nonlocal_declarations, statements to
    # put at the top of the setter; getter_name and setter_name, the names of the new functions.
    # With no variables, the getter returns an empty tuple and the setter does nothing. Otherwise
    # the getter returns a tuple of the variables (composite ones are wrapped in a guarded
    # load that yields an undefined marker when unset) and the setter unpacks a tuple into them.
    # Returns the list of generated function definition nodes.
    if not block_vars:
      template = """
        def getter_name():
          return ()
        def setter_name(block_vars):
          pass
      """
      return templates.replace(
          template, getter_name=getter_name, setter_name=setter_name)

    guarded_block_vars = []
    for v in block_vars:
      if v.is_simple():
        guarded_block_vars.append(v)
      else:
        guarded_block_vars.append(
            templates.replace_as_expression(
                'ag__.ldu(lambda: var_, name)',
                var_=v,
                name=gast.Constant(str(v), kind=None)))

    template = """
      def getter_name():
        return guarded_state_vars,
      def setter_name(vars_):
        nonlocal_declarations
        state_vars, = vars_
    """
    return templates.replace(
        template,
        nonlocal_declarations=nonlocal_declarations,
        getter_name=getter_name,
        guarded_state_vars=guarded_block_vars,
        setter_name=setter_name,
        state_vars=tuple(block_vars))

  def _create_loop_options(self, node):
    # [FUNCTION DOC]
    # HELPER: builds the loop options mapping node passed to the runtime loop operators.
    # If the loop node has directive annotations that include the set-loop-options directive,
    # returns a dictionary node with those option names (as string constants) and their value
    # nodes; otherwise returns an empty dictionary node.
    if not anno.hasanno(node, anno.Basic.DIRECTIVES):
      return gast.Dict([], [])

    loop_directives = anno.getanno(node, anno.Basic.DIRECTIVES)
    if directives.set_loop_options not in loop_directives:
      return gast.Dict([], [])

    opts_dict = loop_directives[directives.set_loop_options]
    str_keys, values = zip(*opts_dict.items())
    keys = [gast.Constant(s, kind=None) for s in str_keys]
    values = list(values)  # ast and gast don't play well with tuples.
    return gast.Dict(keys, values)

  def _create_undefined_assigns(self, undefined_symbols):
    # [FUNCTION DOC]
    # HELPER: for each symbol that is modified in a block but not defined before it, generates an
    # assignment of an autograph undefined placeholder carrying the symbol's source name.
    # Input: iterable of symbols. Returns the list of assignment statement nodes (possibly empty).
    assignments = []
    for s in undefined_symbols:
      template = '''
        var = ag__.Undefined(symbol_name)
      '''
      assignments += templates.replace(
          template,
          var=s,
          symbol_name=gast.Constant(s.ssf(), kind=None))
    return assignments

  def _get_block_basic_vars(self, modified, live_in, live_out):
    # [FUNCTION DOC]
    # HELPER: selects the simple (non-composite) modified symbols that count as block state.
    # Inputs: modified symbols, symbols live into the block and symbols live out of it. A symbol
    # is kept if it is live in, live out, or declared nonlocal in the enclosing function; others
    # are treated as local to the block. Composite symbols are skipped (a TODO notes an error
    # should eventually be raised for them). Returns an immutable set of symbols.
    nonlocals = self.state[_Function].scope.nonlocals
    basic_scope_vars = []
    for s in modified:
      if s.is_composite():
        # TODO(mdan): Raise an error when this happens for a TF scope.
        continue
      # Variables not live into or out of the scope are considered local to the
      # scope.
      if s in live_in or s in live_out or s in nonlocals:
        basic_scope_vars.append(s)
      continue
    return frozenset(basic_scope_vars)

  def _get_block_composite_vars(self, modified, live_in):
    # [FUNCTION DOC]
    # HELPER: selects the modified composite symbols (such as attribute or subscript accesses)
    # that count as block state. A composite is kept only if every symbol in its support set
    # (its parent symbols; literals are ignored) is live into the block, so mutations of objects
    # created inside the block are excluded. Returns an immutable set of symbols.
    # The scope variables corresponding to composite symbols (e.g. `self.x`).
    composite_scope_vars = []
    for s in modified:
      if not s.is_composite():
        continue
      # Mutations made to objects created inside the scope will appear as writes
      # to composite symbols. Because these mutations appear as modifications
      # made to composite symbols, we check whether the composite's parent is
      # actually live into the scope.
      # Example:
      #   while cond:
      #     x = Foo()
      #     x.foo = 2 * x.foo  # x.foo is live into the scope, but x is not.
      #
      # Note that some parents might not be symbols - for example, in x['foo'],
      # 'foo' is a parent, but it's a literal, not a symbol. We don't check the
      # liveness of literals.
      support_set_symbols = tuple(
          sss for sss in s.support_set if sss.is_symbol())
      if not all(sss in live_in for sss in support_set_symbols):
        continue
      composite_scope_vars.append(s)
    return frozenset(composite_scope_vars)

  def _get_block_vars(self, node, modified):
    # [FUNCTION DOC]
    # HELPER: computes the state description of a control flow statement.
    # Inputs: the statement node (carrying defined-in, live-in and live-out annotations) and the
    # set of symbols modified in it. Returns three values: the list of state variables (basic plus
    # composite, with outputs first and input-only variables last, each group sorted), a tuple of
    # non-composite symbols that are modified but not defined beforehand (excluding globals and
    # nonlocals), and the number of output variables (state variables that are not input-only).
    """Determines the variables affected inside a control flow statement."""
    defined_in = anno.getanno(node, anno.Static.DEFINED_VARS_IN)
    live_in = anno.getanno(node, anno.Static.LIVE_VARS_IN)
    live_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)
    fn_scope = self.state[_Function].scope

    basic_scope_vars = self._get_block_basic_vars(
        modified,
        live_in,
        live_out)
    composite_scope_vars = self._get_block_composite_vars(modified, live_in)
    scope_vars = tuple(basic_scope_vars | composite_scope_vars)

    # Variables that are modified inside the scope, but not defined
    # before entering it. Only simple variables must be defined. The
    # composite ones will be implicitly checked at runtime.
    possibly_undefined = (
        modified - defined_in - fn_scope.globals - fn_scope.nonlocals)
    undefined = tuple(v for v in possibly_undefined if not v.is_composite())

    # Variables that are modified inside the scope, and depend on values outside
    # it.
    input_only = basic_scope_vars & live_in - live_out

    # Place the outputs first, then sort lexicographically.
    scope_vars = sorted(scope_vars, key=lambda v: (v in input_only, v))
    nouts = len(scope_vars) - len(input_only)

    return scope_vars, undefined, nouts

  def visit_If(self, node):
    # [FUNCTION DOC]
    # Converts an if statement. After visiting children, computes the state variables modified in
    # the body or else branch, creates undefined-placeholder assignments, nonlocal declarations
    # and state getter/setter functions, and wraps the branches into generated body and else
    # functions (an empty else becomes a pass). Replaces the node with a list of statements that
    # ends in a call to the runtime if-statement operator taking the test, both branch functions,
    # the getter and setter, the state symbol names and the output count. Copies source origin
    # information onto the final call statement. Returns the new list of nodes.
    node = self.generic_visit(node)
    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
    orelse_scope = anno.getanno(node, annos.NodeAnno.ORELSE_SCOPE)

    cond_vars, undefined, nouts = self._get_block_vars(
        node, body_scope.bound | orelse_scope.bound)

    undefined_assigns = self._create_undefined_assigns(undefined)

    nonlocal_declarations = self._create_nonlocal_declarations(cond_vars)

    reserved = body_scope.referenced | orelse_scope.referenced
    state_getter_name = self.ctx.namer.new_symbol('get_state', reserved)
    state_setter_name = self.ctx.namer.new_symbol('set_state', reserved)
    state_functions = self._create_state_functions(
        cond_vars, nonlocal_declarations, state_getter_name, state_setter_name)

    orelse_body = node.orelse
    if not orelse_body:
      orelse_body = [gast.Pass()]

    template = """
      state_functions
      def body_name():
        nonlocal_declarations
        body
      def orelse_name():
        nonlocal_declarations
        orelse
      undefined_assigns
      ag__.if_stmt(
        test,
        body_name,
        orelse_name,
        state_getter_name,
        state_setter_name,
        (symbol_names,),
        nouts)
    """
    new_nodes = templates.replace(
        template,
        body=node.body,
        body_name=self.ctx.namer.new_symbol('if_body', reserved),
        orelse=orelse_body,
        orelse_name=self.ctx.namer.new_symbol('else_body', reserved),
        nonlocal_declarations=nonlocal_declarations,
        nouts=gast.Constant(nouts, kind=None),
        state_functions=state_functions,
        state_getter_name=state_getter_name,
        state_setter_name=state_setter_name,
        symbol_names=tuple(gast.Constant(str(s), kind=None) for s in cond_vars),
        test=node.test,
        undefined_assigns=undefined_assigns)
    origin_info.copy_origin(node, new_nodes[-1])
    return new_nodes

  def visit_While(self, node):
    # [FUNCTION DOC]
    # Converts a while statement. After visiting children, computes loop state variables from the
    # body scope, then generates state getter/setter functions, a loop body function, a loop test
    # function returning the condition, undefined-placeholder assignments, and loop options from
    # any set-loop-options directive. Replaces the node with statements ending in a call to the
    # runtime while-statement operator. Copies source origin onto the final call. Returns the new
    # list of nodes.
    node = self.generic_visit(node)
    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)

    loop_vars, undefined, _ = self._get_block_vars(node, body_scope.bound)

    undefined_assigns = self._create_undefined_assigns(undefined)

    nonlocal_declarations = self._create_nonlocal_declarations(loop_vars)

    reserved = body_scope.referenced
    state_getter_name = self.ctx.namer.new_symbol('get_state', reserved)
    state_setter_name = self.ctx.namer.new_symbol('set_state', reserved)
    state_functions = self._create_state_functions(
        loop_vars, nonlocal_declarations, state_getter_name, state_setter_name)

    opts = self._create_loop_options(node)

    template = """
      state_functions
      def body_name():
        nonlocal_declarations
        body
      def test_name():
        return test
      undefined_assigns
      ag__.while_stmt(
          test_name,
          body_name,
          state_getter_name,
          state_setter_name,
          (symbol_names,),
          opts)
    """
    new_nodes = templates.replace(
        template,
        body=node.body,
        body_name=self.ctx.namer.new_symbol('loop_body', reserved),
        nonlocal_declarations=nonlocal_declarations,
        opts=opts,
        state_functions=state_functions,
        state_getter_name=state_getter_name,
        state_setter_name=state_setter_name,
        symbol_names=tuple(gast.Constant(str(s), kind=None) for s in loop_vars),
        test=node.test,
        test_name=self.ctx.namer.new_symbol('loop_test', reserved),
        undefined_assigns=undefined_assigns)
    origin_info.copy_origin(node, new_nodes[-1])
    return new_nodes

  def visit_For(self, node):
    # [FUNCTION DOC]
    # Converts a for statement. After visiting children, computes loop state variables from the
    # body and iterate scopes, generates state functions, undefined-placeholder assignments, loop
    # options (extended with the unparsed iteration target name), an optional extra-test function
    # when the node has an extra loop test annotation (otherwise none is passed), and a loop body
    # function that takes one iterate argument and unpacks it into the target before the original
    # body. Replaces the node with statements ending in a call to the runtime for-statement
    # operator taking the iterated value. Copies source origin onto generated nodes. Returns the
    # new list of nodes.
    node = self.generic_visit(node)
    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
    iter_scope = anno.getanno(node, annos.NodeAnno.ITERATE_SCOPE)

    loop_vars, undefined, _ = self._get_block_vars(
        node, body_scope.bound | iter_scope.bound)

    undefined_assigns = self._create_undefined_assigns(undefined)

    nonlocal_declarations = self._create_nonlocal_declarations(loop_vars)

    reserved = body_scope.referenced | iter_scope.referenced
    state_getter_name = self.ctx.namer.new_symbol('get_state', reserved)
    state_setter_name = self.ctx.namer.new_symbol('set_state', reserved)
    state_functions = self._create_state_functions(
        loop_vars, nonlocal_declarations, state_getter_name, state_setter_name)

    opts = self._create_loop_options(node)
    opts.keys.append(gast.Constant('iterate_names', kind=None))
    opts.values.append(gast.Constant(
        parser.unparse(node.target, include_encoding_marker=False), kind=None))

    if anno.hasanno(node, anno.Basic.EXTRA_LOOP_TEST):
      extra_test = anno.getanno(node, anno.Basic.EXTRA_LOOP_TEST)
      extra_test_name = self.ctx.namer.new_symbol(
          'extra_test', reserved)
      template = """
        def extra_test_name():
          nonlocal_declarations
          return extra_test_expr
      """
      extra_test_function = templates.replace(
          template,
          extra_test_expr=extra_test,
          extra_test_name=extra_test_name,
          loop_vars=loop_vars,
          nonlocal_declarations=nonlocal_declarations)
    else:
      extra_test_name = parser.parse_expression('None')
      extra_test_function = []

    # iterate_arg_name holds a single arg with the iterates, which may be a
    # tuple.
    iterate_arg_name = self.ctx.namer.new_symbol('itr', reserved)
    template = """
      iterates = iterate_arg_name
    """
    iterate_expansion = templates.replace(
        template, iterate_arg_name=iterate_arg_name, iterates=node.target)
    origin_info.copy_origin(node, iterate_expansion)

    template = """
      state_functions
      def body_name(iterate_arg_name):
        nonlocal_declarations
        iterate_expansion
        body
      extra_test_function
      undefined_assigns
      ag__.for_stmt(
          iterated,
          extra_test_name,
          body_name,
          state_getter_name,
          state_setter_name,
          (symbol_names,),
          opts)
    """
    new_nodes = templates.replace(
        template,
        body=node.body,
        body_name=self.ctx.namer.new_symbol('loop_body', reserved),
        extra_test_function=extra_test_function,
        extra_test_name=extra_test_name,
        iterate_arg_name=iterate_arg_name,
        iterate_expansion=iterate_expansion,
        iterated=node.iter,
        nonlocal_declarations=nonlocal_declarations,
        opts=opts,
        symbol_names=tuple(gast.Constant(str(s), kind=None) for s in loop_vars),
        state_functions=state_functions,
        state_getter_name=state_getter_name,
        state_setter_name=state_setter_name,
        undefined_assigns=undefined_assigns)
    origin_info.copy_origin(node, new_nodes[-1])
    return new_nodes


class AnnotatedDef(reaching_definitions.Definition):
  # [CLASS DOC]
  # Reaching-definitions Definition subclass that additionally carries a mapping of directives
  # (initially empty). KNOWN ISSUE: it is not referenced elsewhere in this module, so it appears
  # to be unused here.

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor: calls the parent constructor and initializes an empty directives mapping.
    super(AnnotatedDef, self).__init__()
    self.directives = {}


def transform(node, ctx):
  # [FUNCTION DOC]
  # Entry point of the converter. Inputs: an AST node (typically a function definition) and the
  # converter context. Runs the analyses in order (control flow graphs, qualified names,
  # activity, reaching definitions, reaching function definitions, liveness), then applies
  # ControlFlowTransformer. Returns the transformed node.
  graphs = cfg.build(node)
  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx, None)
  node = reaching_definitions.resolve(node, ctx, graphs)
  node = reaching_fndefs.resolve(node, ctx, graphs)
  node = liveness.resolve(node, ctx, graphs)

  node = ControlFlowTransformer(ctx).visit(node)
  return node
