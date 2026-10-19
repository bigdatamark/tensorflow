# [MODULE DOC]
# AutoGraph converter that rewrites assert statements into calls to the runtime assert function
# so they can be staged as graph operations. Exposes the AssertTransformer node transformer and
# the transform entry point used by the conversion pipeline.
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
"""Converts assert statements to their corresponding TF calls."""

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import templates


class AssertTransformer(converter.Base):
  # [CLASS DOC]
  # Syntax-tree transformer (subclass of the converter base) that replaces assert statements with
  # a call to the runtime assert helper taking the test and a deferred message. Created with a
  # conversion context by the transform function.
  """Transforms Assert nodes to Call so they can be handled as functions."""

  def visit_Assert(self, node):
    # [FUNCTION DOC]
    # Visitor for assert nodes. First transforms children, then returns replacement statements
    # calling the runtime assert helper with the test and a deferred message. A missing message
    # becomes the constant text "Assertion error"; a constant message is used as is; any other
    # message expression raises a not-implemented error (only string messages supported).
    self.generic_visit(node)

    # Note: The lone tf.Assert call will be wrapped with control_dependencies
    # by side_effect_guards.
    template = """
      ag__.assert_stmt(test, lambda: msg)
    """

    if node.msg is None:
      return templates.replace(
          template,
          test=node.test,
          msg=gast.Constant('Assertion error', kind=None))
    elif isinstance(node.msg, gast.Constant):
      return templates.replace(template, test=node.test, msg=node.msg)
    else:
      raise NotImplementedError('can only convert string messages for now.')


def transform(node, ctx):
  # [FUNCTION DOC]
  # Converter entry point. Inputs: node (syntax tree to convert) and ctx (conversion context).
  # Returns the tree with assert statements rewritten.
  node = AssertTransformer(ctx).visit(node)
  return node
