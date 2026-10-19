# [MODULE DOC]
# Autograph converter that finds calls to autograph directive functions (set element type and set
# loop options), removes those calls from the generated code and records their arguments as AST
# annotations (on definitions, or on the enclosing loop node). Also annotates names and attribute
# accesses with their statically known values. Exposes the transform entry function and the
# STATIC_VALUE annotation key. No import-time side effects.
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
"""Handles directives.

This converter removes the directive functions from the code and moves the
information they specify into AST annotations. It is a specialized form of
static analysis, one that is specific to AutoGraph.

Note that this requires that the actual directive functions are static - that
is, they do not change at runtime. So if you do something like this:

  tf.autograph.set_loop_options = <new function>

Then the directive will may no longer be recognized. Furthermore, if the
converted function is cached, such an action may be irreversible.
"""

import inspect

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.util import tf_inspect


STATIC_VALUE = 'static_value'
"""Used for AST annotations, see visit_Name."""


class _LoopScope(object):
  # [CLASS DOC]
  # HELPER: state record for DirectivesTransformer tracking the loop being visited. Attributes:
  # the loop AST node and the count of statements visited in it. KNOWN ISSUE: the transformer
  # also uses enter, exit and level on this state, which are not defined here; they presumably come
  # from the converter state stack machinery.

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor: sets the loop node to nothing and the visited statement count to zero.
    self.ast_node = None
    self.statements_visited = 0


def _map_args(call_node, function):
  # [FUNCTION DOC]
  # HELPER: maps the arguments of a directive call node to the directive function parameters.
  # Inputs: the call AST node and the actual directive function. Returns a mapping from parameter
  # names to argument AST nodes, dropping parameters left at the unspecified marker default.
  # Raises a value error if a parameter not supplied by the call has a default other than the
  # unspecified marker. KNOWN ISSUE: the error message formats a lazy pairing object rather than
  # a readable list, so its content is not shown usefully.
  """Maps AST call nodes to the actual function's arguments.

  Args:
    call_node: ast.Call
    function: Callable[..., Any], the actual function matching call_node
  Returns:
    Dict[Text, ast.AST], mapping each of the function's argument names to
    the respective AST node.
  Raises:
      ValueError: if the default arguments are not correctly set
  """
  args = call_node.args
  kwds = {kwd.arg: kwd.value for kwd in call_node.keywords}
  call_args = tf_inspect.getcallargs(function, *args, **kwds)

  # Keyword arguments not specified in kwds will be mapped to their defaults,
  # which are Python values. Since we don't currently have a way to transform
  # those into AST references, we simply remove them. By convention, directives
  # use UNSPECIFIED as default value for optional arguments. No other
  # defaults should be present.
  unexpected_defaults = []
  for k in call_args:
    if (k not in kwds
        and call_args[k] not in args
        and call_args[k] is not directives.UNSPECIFIED):
      unexpected_defaults.append(k)
  if unexpected_defaults:
    raise ValueError('Unexpected keyword argument values, %s, for function %s'
                     % (zip(unexpected_defaults,
                            [call_args[k] for k in unexpected_defaults]),
                        function))
  return {k: v for k, v in call_args.items() if v is not directives.UNSPECIFIED}


class DirectivesTransformer(converter.Base):
  # [CLASS DOC]
  # AST transformer implementing the directives converter. It annotates statically known names,
  # drops directive call statements and stores their parameters either on the symbol definitions
  # (element type directive) or on the enclosing loop node (loop options directive). Tracks loop
  # nesting with a loop scope state.
  """Parses compiler directives and converts them into AST annotations."""

  def _process_symbol_directive(self, call_node, directive):
    # [FUNCTION DOC]
    # HELPER: handles a directive that targets a symbol. Takes the call node and the directive
    # function. Raises a value error if the call has no positional first argument. Otherwise stores
    # the mapped arguments in the directives mapping of every original definition of the first
    # argument (mutating those definitions) and returns the call node.
    if len(call_node.args) < 1:
      raise ValueError('"%s" requires a positional first argument'
                       ' as the target' % directive.__name__)
    target = call_node.args[0]
    defs = anno.getanno(target, anno.Static.ORIG_DEFINITIONS)
    for def_ in defs:
      def_.directives[directive] = _map_args(call_node, directive)
    return call_node

  def _process_statement_directive(self, call_node, directive):
    # [FUNCTION DOC]
    # HELPER: handles a directive that applies to the enclosing loop. Raises a value error if more
    # than one statement of the loop was already visited (not first statement) or if the nesting
    # level is below two (not inside a statement). Otherwise merges the mapped arguments into the
    # directives annotation of the loop node and returns the call node.
    if self.state[_LoopScope].statements_visited > 1:
      raise ValueError(
          '"%s" must be the first statement in the loop block' % (
              directive.__name__))
    if self.state[_LoopScope].level < 2:
      raise ValueError(
          '"%s" must be used inside a statement' % directive.__name__)
    target = self.state[_LoopScope].ast_node
    node_anno = anno.getanno(target, anno.Basic.DIRECTIVES, {})
    node_anno[directive] = _map_args(call_node, directive)
    anno.setanno(target, anno.Basic.DIRECTIVES, node_anno)
    return call_node

  def visit_Name(self, node):
    # [FUNCTION DOC]
    # Visits a name node. For a loaded name with no local definitions that exists in the function
    # namespace, annotates it with its static value. Returns the node.
    node = self.generic_visit(node)
    if isinstance(node.ctx, gast.Load):
      defs = anno.getanno(node, anno.Static.DEFINITIONS, ())
      is_defined = bool(defs)
      if not is_defined and node.id in self.ctx.info.namespace:
        anno.setanno(node, STATIC_VALUE, self.ctx.info.namespace[node.id])
    return node

  def visit_Attribute(self, node):
    # [FUNCTION DOC]
    # Visits an attribute node. If the parent has a static value that is a module having the
    # attribute, annotates the node with that attribute's static value. Returns the node.
    node = self.generic_visit(node)
    parent_val = anno.getanno(node.value, STATIC_VALUE, default=None)
    if parent_val is not None and inspect.ismodule(parent_val):
      if hasattr(parent_val, node.attr):
        anno.setanno(node, STATIC_VALUE, getattr(parent_val, node.attr))
    return node

  def visit_Assign(self, node):
    # [FUNCTION DOC]
    # Counts the statement for the current loop scope, then visits children and returns the result.
    self.state[_LoopScope].statements_visited += 1
    return self.generic_visit(node)

  def visit_AugAssign(self, node):
    # [FUNCTION DOC]
    # Counts the statement for the current loop scope, then visits children and returns the result.
    self.state[_LoopScope].statements_visited += 1
    return self.generic_visit(node)

  def visit_Expr(self, node):
    # [FUNCTION DOC]
    # Counts the statement, visits children, and if the expression is a call whose function has a
    # static value equal to the element type directive or the loop options directive, processes it
    # and returns nothing so the call is removed from the output. Otherwise returns the node.
    self.state[_LoopScope].statements_visited += 1
    node = self.generic_visit(node)
    if isinstance(node.value, gast.Call):
      call_node = node.value
      static_val = anno.getanno(call_node.func, STATIC_VALUE, default=None)
      if static_val is not None:
        # Note: directive calls are not output in the generated code, hence
        # the removal from the code by returning None.

        if static_val is directives.set_element_type:
          self._process_symbol_directive(call_node, static_val)
          return None
        elif static_val is directives.set_loop_options:
          self._process_statement_directive(call_node, static_val)
          return None
    return node

  # TODO(mdan): This will be insufficient for other control flow.
  # That means that if we ever have a directive that affects things other than
  # loops, we'll need support for parallel scopes, or have multiple converters.
  def _track_and_visit_loop(self, node):
    # [FUNCTION DOC]
    # HELPER: shared handling for loops. Enters a loop scope, records the node, visits children,
    # replaces an emptied body with a pass statement, exits the scope and returns the node.
    self.state[_LoopScope].enter()
    self.state[_LoopScope].ast_node = node
    node = self.generic_visit(node)
    # Edge case: a loop with just one directive statement would become empty.
    if not node.body:
      node.body = [gast.Pass()]
    self.state[_LoopScope].exit()
    return node

  def visit_While(self, node):
    # [FUNCTION DOC]
    # Handles a while loop by delegating to the shared loop tracking helper; returns its result.
    return self._track_and_visit_loop(node)

  def visit_For(self, node):
    # [FUNCTION DOC]
    # Handles a for loop by delegating to the shared loop tracking helper; returns its result.
    return self._track_and_visit_loop(node)


def transform(node, ctx):
  # [FUNCTION DOC]
  # Entry point: applies DirectivesTransformer with the given converter context to the AST node
  # and returns the transformed node.
  return DirectivesTransformer(ctx).visit(node)
