# [MODULE DOC]
# Autograph converter that rewrites logical expressions (and, or, not, chained comparisons and,
# when the feature is enabled, equality and inequality) into calls to autograph runtime operators
# so they work with tensors. Boolean operands are wrapped in lambdas to preserve lazy evaluation.
# Exposes the operator mapping tables, a safe operand marker constant and the transform entry
# function. No import-time side effects.
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
"""Converter for logical expressions, e.g. `a and b -> tf.logical_and(a, b)`."""

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates

# TODO(mdan): Properly extract boolean ops according to lazy eval rules.
# Note that this isn't completely safe either, because tensors may have control
# dependencies.
# Note that for loops that should be done after the loop was converted to
# tf.while_loop so that the expanded conditionals are properly scoped.

# Used to signal that an operand is safe for non-lazy evaluation.
SAFE_BOOLEAN_OPERAND = 'SAFE_BOOLEAN_OPERAND'


LOGICAL_OPERATORS = {
    gast.And: 'ag__.and_',
    gast.Not: 'ag__.not_',
    gast.Or: 'ag__.or_',
}

EQUALITY_OPERATORS = {
    gast.Eq: 'ag__.eq',
    gast.NotEq: 'ag__.not_eq',
}


class LogicalExpressionTransformer(converter.Base):
  # [CLASS DOC]
  # AST transformer converting boolean operations, unary not, and comparisons into calls to the
  # runtime logical operator functions. Equality operators are only overloaded when the equality
  # operators feature is enabled in the user options.
  """Converts logical expressions to corresponding TF calls."""

  def _overload_of(self, operator):
    # [FUNCTION DOC]
    # HELPER: looks up the runtime function name for an operator node. Logical operators always map
    # to their runtime names; equality operators map only if the equality feature is enabled.
    # Returns the name as text, or nothing when the operator should not be overloaded.
    op_type = type(operator)
    if op_type in LOGICAL_OPERATORS:
      return LOGICAL_OPERATORS[op_type]
    if self.ctx.user.options.uses(converter.Feature.EQUALITY_OPERATORS):
      if op_type in EQUALITY_OPERATORS:
        return EQUALITY_OPERATORS[op_type]
    return None

  def _as_lambda(self, expr):
    # [FUNCTION DOC]
    # HELPER: wraps an expression node in a lambda with no parameters (for lazy evaluation).
    # Returns the new expression node.
    return templates.replace_as_expression('lambda: expr', expr=expr)

  def _as_binary_function(self, func_name, arg1, arg2):
    # [FUNCTION DOC]
    # HELPER: builds a call expression to the named function with two arguments. func_name is a
    # dotted name as text. Returns the call expression node.
    return templates.replace_as_expression(
        'func_name(arg1, arg2)',
        func_name=parser.parse_expression(func_name),
        arg1=arg1,
        arg2=arg2)

  def _as_binary_operation(self, op, arg1, arg2):
    # [FUNCTION DOC]
    # HELPER: builds a binary comparison expression of arg1 and arg2 using the given original
    # operator node (built from an identity comparison whose operator is then replaced). Returns the
    # comparison node.
    template = templates.replace_as_expression(
        'arg1 is arg2',  # Note: `is` will be replaced with `op` below.
        arg1=arg1,
        arg2=arg2)
    template.ops[0] = op
    return template

  def _as_unary_function(self, func_name, arg):
    # [FUNCTION DOC]
    # HELPER: builds a call expression to the named function with one argument. Returns the call
    # node.
    return templates.replace_as_expression(
        'func_name(arg)', func_name=parser.parse_expression(func_name), arg=arg)

  def _process_binop(self, op, left, right):
    # [FUNCTION DOC]
    # HELPER: converts one binary comparison. If the operator has a runtime overload returns a call
    # to it with the two operands; otherwise rebuilds the original comparison. Returns the node.
    overload = self._overload_of(op)
    if overload is None:
      return self._as_binary_operation(op, left, right)
    return self._as_binary_function(overload, left, right)

  def visit_Compare(self, node):
    # [FUNCTION DOC]
    # Converts a comparison. Visits children, then splits chained comparisons into pairs and
    # converts each; consecutive pairs are joined by the runtime and operator with each side wrapped
    # in a lambda, so a chain of comparisons becomes a conjunction. Returns the resulting
    # expression node. Asserts that at least one comparison exists.
    node = self.generic_visit(node)

    ops_and_comps = list(zip(node.ops, node.comparators))
    left = node.left

    # Repeated comparisons are converted to conjunctions:
    #   a < b < c   ->   a < b and b < c
    op_tree = None
    while ops_and_comps:
      op, right = ops_and_comps.pop(0)
      binary_comparison = self._process_binop(op, left, right)
      if op_tree is not None:
        op_tree = self._as_binary_function('ag__.and_',
                                           self._as_lambda(op_tree),
                                           self._as_lambda(binary_comparison))
      else:
        op_tree = binary_comparison
      left = right

    assert op_tree is not None
    return op_tree

  def visit_UnaryOp(self, node):
    # [FUNCTION DOC]
    # Converts a unary operation. Visits children; if the operator has an overload (not), returns
    # a call to the runtime function with the operand; otherwise returns the node unchanged.
    node = self.generic_visit(node)

    overload = self._overload_of(node.op)
    if overload is None:
      return node

    return self._as_unary_function(overload, node.operand)

  def visit_BoolOp(self, node):
    # [FUNCTION DOC]
    # Converts and/or expressions. Visits children, then folds the operand list from the right: the
    # last operand is combined with the previous one through the runtime and or or function, each
    # side wrapped in a lambda, repeating until all operands are used. Mutates the node's operand
    # list (it is emptied). Returns the nested call expression.
    node = self.generic_visit(node)
    node_values = node.values
    right = node.values.pop()
    while node_values:
      left = node_values.pop()
      right = self._as_binary_function(
          self._overload_of(node.op), self._as_lambda(left),
          self._as_lambda(right))
    return right


def transform(node, ctx):
  # [FUNCTION DOC]
  # Entry point: applies LogicalExpressionTransformer with the given context to the AST node and
  # returns the transformed node.
  transformer = LogicalExpressionTransformer(ctx)
  return transformer.visit(node)
