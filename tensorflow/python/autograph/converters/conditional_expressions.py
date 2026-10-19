# [MODULE DOC]
# AutoGraph converter for the ternary conditional expression. Rewrites each such expression into
# a call to the runtime conditional-expression function with the test, deferred true and false
# branches (as parameterless lambdas) and the source text of the test. Exposes
# ConditionalExpressionTransformer and transform.
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
"""Converts the ternary conditional operator."""

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates


class ConditionalExpressionTransformer(converter.Base):
  # [CLASS DOC]
  # Syntax-tree transformer (subclass of the converter base) turning ternary conditional
  # expressions into runtime function calls. Created with a conversion context by transform.
  """Converts conditional expressions to functional form."""

  def visit_IfExp(self, node):
    # [FUNCTION DOC]
    # Visitor for a conditional expression node. Returns an expression calling the runtime
    # conditional-expression function with the test, the true branch and false branch each wrapped
    # as a deferred parameterless function, and a string constant holding the source text of the
    # test (for error messages). Note: children are not visited separately, so nested conditional
    # expressions are handled only through the templates, not by recursion here.
    template = '''
        ag__.if_exp(
            test,
            lambda: true_expr,
            lambda: false_expr,
            expr_repr)
    '''
    expr_repr = parser.unparse(node.test, include_encoding_marker=False).strip()
    return templates.replace_as_expression(
        template,
        test=node.test,
        true_expr=node.body,
        false_expr=node.orelse,
        expr_repr=gast.Constant(expr_repr, kind=None))


def transform(node, ctx):
  # [FUNCTION DOC]
  # Converter entry point. Inputs: node (syntax tree) and ctx (conversion context). Returns the
  # tree with conditional expressions converted.
  node = ConditionalExpressionTransformer(ctx).visit(node)
  return node
