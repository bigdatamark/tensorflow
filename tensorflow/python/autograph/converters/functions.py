# [MODULE DOC]
# Autograph converter that wraps converted function definitions and lambdas in the runtime function
# scope boilerplate (a context manager that carries conversion options and the function name). It
# also decorates nested functions so they are not converted twice and strips decorators from the
# top-level function. Exposes the transform entry function. No import-time side effects.
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
"""Converts function definitions and lambdas by adding necessary boilerplate."""

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos


class _Function(object):
  # [CLASS DOC]
  # HELPER: per-function state record for FunctionTransformer. Holds the generated name of the
  # function context variable (initially nothing). Used through the transformer state stack.

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor: sets the context name to nothing.
    self.context_name = None


class FunctionTransformer(converter.Base):
  # [CLASS DOC]
  # AST transformer that wraps function bodies in the autograph function scope and lambda bodies
  # in a function scope call. Tracks nesting depth through a state stack of function records.
  """Wraps function bodies around autograph-specific boilerplate."""

  def _function_scope_options(self, fn_scope):
    # [FUNCTION DOC]
    # HELPER: chooses conversion options for a generated function scope. Takes the function state
    # record. At nesting level 2 (the top-level function) returns the user requested options;
    # otherwise returns the options for a recursive call conversion.
    """Returns the options with which to create function scopes."""
    # Top-level function receive the options that were directly requested.
    # All others receive the options corresponding to a recursive conversion.
    # Note: this mainly controls the user_requested flag, which is important
    # primarily because the FunctionScope context also creates a
    # ControlStatusCtx(autograph=ENABLED) when user_requested is True. See
    # function_wrappers.py.
    if fn_scope.level == 2:
      return self.ctx.user.options
    return self.ctx.user.options.call_options()

  def visit_Lambda(self, node):
    # [FUNCTION DOC]
    # Converts a lambda. Visits children first; if nested deeper than level 2, wraps the lambda in
    # the artifact marking call and returns that expression. Otherwise creates a unique context
    # name, records it on the state and as an annotation, and replaces the lambda body with a call
    # to the runtime function-scope helper holding the options. Returns the lambda node.
    with self.state[_Function] as fn_scope:
      node = self.generic_visit(node)

      # TODO(mdan): Fix the tests so that we can always add this decorator.
      if fn_scope.level > 2:
        return templates.replace_as_expression(
            'ag__.autograph_artifact(l)', l=node)

      scope = anno.getanno(node, anno.Static.SCOPE)
      function_context_name = self.ctx.namer.new_symbol('lscope',
                                                        scope.referenced)
      fn_scope.context_name = function_context_name
      anno.setanno(node, 'function_context_name', function_context_name)

      template = """
        ag__.with_function_scope(
            lambda function_context: body, function_context_name, options)
      """
      node.body = templates.replace_as_expression(
          template,
          options=self._function_scope_options(fn_scope).to_ast(),
          function_context=function_context_name,
          function_context_name=gast.Constant(function_context_name, kind=None),
          body=node.body)

      return node

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Converts a function definition. Creates a unique context name (recorded as state and
    # annotation), visits children, then clears decorators for top-level functions (level 2 or
    # lower) or appends the artifact decorator for inner ones to avoid double conversion. Any
    # leading docstring is kept out of the wrapped body; the remaining body is placed inside a
    # with-block using the runtime function scope (given function name, context name, options), with
    # the docstring put back first. Returns the modified node.
    with self.state[_Function] as fn_scope:
      scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)

      function_context_name = self.ctx.namer.new_symbol('fscope',
                                                        scope.referenced)
      fn_scope.context_name = function_context_name
      anno.setanno(node, 'function_context_name', function_context_name)

      node = self.generic_visit(node)

      if fn_scope.level <= 2:
        # Top-level functions lose their decorator because the conversion is
        # always just-in-time and by the time it happens the decorators are
        # already set to be applied.
        node.decorator_list = []
      else:
        # TODO(mdan): Fix the tests so that we can always add this decorator.
        # Inner functions are converted already, so we insert a decorator to
        # prevent double conversion. Double conversion would work too, but this
        # saves the overhead.
        node.decorator_list.append(
            parser.parse_expression('ag__.autograph_artifact'))

      docstring_node = None
      if node.body:
        first_statement = node.body[0]
        if (isinstance(first_statement, gast.Expr) and
            isinstance(first_statement.value, gast.Constant)):
          docstring_node = first_statement
          node.body = node.body[1:]

      template = """
        with ag__.FunctionScope(
            function_name, context_name, options) as function_context:
          body
      """
      wrapped_body = templates.replace(
          template,
          function_name=gast.Constant(node.name, kind=None),
          context_name=gast.Constant(function_context_name, kind=None),
          options=self._function_scope_options(fn_scope).to_ast(),
          function_context=function_context_name,
          body=node.body)

      if docstring_node is not None:
        wrapped_body = [docstring_node] + wrapped_body

      node.body = wrapped_body

      return node


def transform(node, ctx):
  # [FUNCTION DOC]
  # Entry point: resolves qualified names and activity analysis for the node, then applies
  # FunctionTransformer with the given context. Returns the transformed node.
  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx, None)

  return FunctionTransformer(ctx).visit(node)
