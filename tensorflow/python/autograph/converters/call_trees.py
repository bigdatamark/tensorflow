# [MODULE DOC]
# AutoGraph converter for function calls. Rewrites each call in converted code into a call to the
# runtime converted-call function, passing the callee, a tuple of positional arguments, a
# dictionary (or null) of keyword arguments and the enclosing function's context name. Calls to
# the internal AutoGraph module, to the function-scope context manager, to debugger break
# functions and (unless the builtin-functions feature is on) to print are left alone. Requires the
# function-scopes converter to have run. Exposes CallTreeTransformer and transform. Module state:
# a flag recording whether the debugger warning was already logged.
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
"""Handles function calls, by generating compiled function names and calls.

Note: this transformer does not rename the top level object being converted;
that is the caller's responsibility.

Requires function_scopes.
"""

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.utils import ag_logging


# TODO(mdan): Rename to FunctionCallsTransformer.


class _Function(object):
  # [CLASS DOC]
  # HELPER: per-function state record for CallTreeTransformer, kept on its state stack. Holds the
  # name of the function-context variable (null until set). Class attribute marks that the state
  # stack has no root entry.

  no_root = True

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the record with the context name unset (null).
    self.context_name = None


set_trace_warned = False


class _ArgTemplateBuilder(object):
  # [CLASS DOC]
  # HELPER: used by CallTreeTransformer to build one tuple expression out of a call's positional
  # arguments, including starred ones.
  # Holds a list of pending plain arguments, a list of completed pieces and a finalized flag.
  # Usage: add plain or starred arguments in order, call finalize, then ask for the tree.
  """Constructs a tuple representing the positional arguments in a call.

  Example (yes, it's legal Python 3):

      f(*args1, b, *args2, c, d)  ->  args1 + (b,) + args2 + (c, d)
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes empty pending arguments, empty pieces list and not-finalized state.
    self._arg_accumulator = []
    self._argspec = []
    self._finalized = False

  def _consume_args(self):
    # [FUNCTION DOC]
    # HELPER: flushes pending plain arguments into one tuple piece (if any) and clears them.
    if self._arg_accumulator:
      self._argspec.append(
          gast.Tuple(elts=self._arg_accumulator, ctx=gast.Load()))
      self._arg_accumulator = []

  def add_arg(self, a):
    # [FUNCTION DOC]
    # Queues a plain positional argument expression (a). Returns nothing.
    self._arg_accumulator.append(a)

  def add_stararg(self, a):
    # [FUNCTION DOC]
    # Adds a starred argument. First flushes pending plain arguments, then appends a piece that
    # converts expression a to a tuple via the tuple builtin. Returns nothing.
    self._consume_args()
    self._argspec.append(
        gast.Call(
            gast.Name(
                'tuple', ctx=gast.Load(), annotation=None, type_comment=None),
            args=[a],
            keywords=()))

  def finalize(self):
    # [FUNCTION DOC]
    # Flushes any pending plain arguments and marks the builder complete. Returns nothing.
    self._consume_args()
    self._finalized = True

  def to_ast(self):
    # [FUNCTION DOC]
    # Returns the syntax tree of the combined arguments: the pieces concatenated with addition, or
    # an empty tuple if there were no arguments. Asserts that finalize was called first.
    assert self._finalized
    if self._argspec:
      result = self._argspec[0]
      for i in range(1, len(self._argspec)):
        result = gast.BinOp(result, gast.Add(), self._argspec[i])
      return result
    return gast.Tuple([], gast.Load())


class CallTreeTransformer(converter.Base):
  # [CLASS DOC]
  # Syntax-tree transformer (subclass of the converter base) that wraps function calls in the
  # runtime converted-call function. Tracks the current function context name on a state stack
  # (pushed for function definitions and for lambdas that have a context annotation). Created with
  # a conversion context by transform.
  """Transforms the call tree by renaming transformed symbols."""

  def visit_Lambda(self, node):
    # [FUNCTION DOC]
    # Visitor for lambdas. If the lambda has no function-context annotation (lambdas generated
    # during conversion), just visits children. Otherwise pushes a function state carrying the
    # annotated context name while visiting children. Returns the visited node.
    if not anno.hasanno(node, 'function_context_name'):
      # Lambda functions created during the conversion process have no
      # context manager.
      return self.generic_visit(node)
    with self.state[_Function] as fn_scope:
      fn_scope.context_name = anno.getanno(node, 'function_context_name')
      return self.generic_visit(node)

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Visitor for function definitions. Decorators, argument defaults and keyword defaults are
    # visited in the outer scope. Then pushes a function state with the context name annotation
    # (asserts it exists, since the function-scopes converter always adds one), visits the body
    # and the return annotation. Returns the node.
    # Decorators and arg defaults are part of the outer scope.
    node.decorator_list = self.visit_block(node.decorator_list)
    node.args.defaults = self.visit_block(node.args.defaults)
    for i, d in enumerate(node.args.kw_defaults):
      if d is not None:
        node.args.kw_defaults[i] = self.visit(d)
    with self.state[_Function] as fn_scope:
      # Note: if the conversion process ever creates helper functions, this
      # assumption will no longer hold.
      assert anno.hasanno(node, 'function_context_name'), (
          'The function_scopes converter always creates a scope for functions.')
      fn_scope.context_name = anno.getanno(node, 'function_context_name')
      node.body = self.visit_block(node.body)
      if node.returns:
        node.returns = self.visit(node.returns)
      return node

  def visit_With(self, node):
    # [FUNCTION DOC]
    # Visitor for with statements. Only the body is converted; the context manager expressions are
    # left unconverted. Returns the node.
    # Context manager calls (in node.items) are not converted.
    node.body = self.visit_block(node.body)
    return node

  def _args_to_tuple(self, node):
    # [FUNCTION DOC]
    # HELPER: builds the positional-arguments expression for visit_Call.
    # Input: a call node. Returns a tree for a single tuple combining plain and starred
    # arguments in order.
    """Ties together all positional and *arg arguments in a single tuple."""
    # TODO(mdan): We could rewrite this to just a call to tuple(). Maybe better?
    # For example for
    #   f(a, b, *args)
    # instead of writing:
    #   (a, b) + args
    # just write this?
    #   tuple(a, b, *args)
    builder = _ArgTemplateBuilder()
    for a in node.args:
      if isinstance(a, gast.Starred):
        builder.add_stararg(a.value)
      else:
        builder.add_arg(a)
    builder.finalize()
    return builder.to_ast()

  def _kwargs_to_dict(self, node):
    # [FUNCTION DOC]
    # HELPER: builds the keyword-arguments expression for visit_Call.
    # Input: a call node. Returns a tree for a dictionary built from its keywords (including
    # double-starred ones), or a tree for the null value if there are no keywords.
    """Ties together all keyword and **kwarg arguments in a single dict."""
    if node.keywords:
      return gast.Call(
          gast.Name(
              'dict', ctx=gast.Load(), annotation=None, type_comment=None),
          args=(),
          keywords=node.keywords)
    else:
      return parser.parse_expression('None')

  def visit_Call(self, node):
    # [FUNCTION DOC]
    # Visitor for call nodes. Visits children first. Returns the node unchanged for calls whose
    # qualified name starts with the internal AutoGraph module prefix or the current function
    # context prefix; for debugger break functions (logging a one-time warning, tracked in a
    # module-level flag, side effect); and for print when the builtin-functions feature is not
    # enabled. Otherwise returns an expression calling the runtime converted-call function with the
    # callee, argument tuple, keyword dictionary (or null) and function context name.
    full_name = str(anno.getanno(node.func, anno.Basic.QN, default=''))
    function_context_name = self.state[_Function].context_name
    node = self.generic_visit(node)

    # TODO(mdan): Refactor converted_call as a 'Call' operator.

    # Calls to the internal 'ag__' module are never converted (though their
    # arguments might be).
    if full_name.startswith('ag__.'):
      return node

    # Calls to the function context manager (inserted by function_scopes) are
    # also safe.
    if full_name.startswith(function_context_name + '.'):
      return node

    # Calls to pdb.set_trace or ipdb.set_trace are never converted. We don't use
    # the normal mechanisms to bypass these literals because they are sensitive
    # to the frame they are being called from.
    # TODO(mdan): Generalize this to a "static allowlist" config.
    if full_name in ('pdb.set_trace', 'ipdb.set_trace', 'breakpoint'):
      global set_trace_warned
      if not set_trace_warned:
        # TODO(mdan): Update and shorten once available on tensorflow.org.
        ag_logging.warning(
            'Detected `pdb.set_trace()` in user code. The code'
            ' generated by AutoGraph is not optimized for step-by-step'
            ' debugging. See https://github.com/tensorflow/tensorflow/'
            'blob/master/tensorflow/python/autograph/g3doc/reference/'
            'debugging.md.')
        set_trace_warned = True
      return node

    if (full_name == 'print' and
        not self.ctx.user.options.uses(converter.Feature.BUILTIN_FUNCTIONS)):
      return node

    template = """
      ag__.converted_call(func, args, kwargs, function_ctx)
    """
    new_call = templates.replace_as_expression(
        template,
        func=node.func,
        args=self._args_to_tuple(node),
        kwargs=self._kwargs_to_dict(node),
        function_ctx=function_context_name)

    return new_call


def transform(node, ctx):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring claims a (node, new_names) pair is returned, but the code returns
  # only the transformed tree.
  # Converter entry point. Inputs: node (syntax tree) and ctx (conversion context). Resolves
  # qualified names, then applies the call-tree transformer. Returns the converted tree.
  """Transform function call to the compiled counterparts.

  Args:
    node: AST
    ctx: EntityContext
  Returns:
    A tuple (node, new_names):
        node: The transformed AST
        new_names: set(string), containing any newly-generated names
  """
  node = qual_names.resolve(node)

  node = CallTreeTransformer(ctx).visit(node)
  return node
