# [MODULE DOC]
# AutoGraph converter for slice (subscript) operations. Rewrites single-index reads into calls to
# the AutoGraph get-item runtime function (passing an element dtype option taken from a set
# element type directive, or nothing) and single-index assignments into calls to the set-item
# function, so that tensor arrays and tensor lists, which need dedicated read/write functions,
# work. Slices and tuple indices are left alone. Exposes a transformer class and a transform
# entry function. No import-time side effects.
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
"""Converter for slice operations."""

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import templates


class SliceTransformer(converter.Base):
  # [CLASS DOC]
  # AST transformer converting subscript reads and single assignments to subscripts into runtime
  # get-item and set-item calls. Created with the transformation context by the transform
  # function; subclass of the converter base transformer. Holds no extra state.
  """Converts slicing operations to their TF counterpart.

  Currently, relying on the default slice operator that Tensor uses is
  insufficient, because TensorArray and tensor lists use dedicated index read
  and write functions.
  """

  def _process_single_assignment(self, target, value):
    # [FUNCTION DOC]
    # HELPER: builds the replacement for one assignment target in the assign visitor.
    # Inputs: the assignment target node and the assigned value node. If the target is not a
    # subscript, or its index is a tuple or a slice, returns nothing (no replacement). Otherwise
    # returns the statement that assigns the result of the set-item runtime call (container,
    # key, value) back to the container.
    if not isinstance(target, gast.Subscript):
      return None
    s = target.slice
    if isinstance(s, (gast.Tuple, gast.Slice)):
      return None

    template = """
      target = ag__.set_item(target, key, item)
    """
    return templates.replace(
        template, target=target.value, key=target.slice, item=value)

  def visit_Assign(self, node):
    # [FUNCTION DOC]
    # Visitor for assignments. Visits children first, then rewrites a subscript-target assignment
    # via the single assignment helper. Raises a not-implemented error when there is more than
    # one target (multiple assignment); unpacking is not supported. Returns the replacement
    # statements if one was produced, otherwise the (visited) node.
    node = self.generic_visit(node)
    # TODO(mdan): Support unpackings and multiple assignments.
    if len(node.targets) != 1:
      raise NotImplementedError('multiple assignment')
    replacement = self._process_single_assignment(node.targets[0], node.value)
    if replacement is not None:
      return replacement
    return node

  def visit_Subscript(self, node):
    # [FUNCTION DOC]
    # Visitor for subscript expressions. Visits children, then leaves the node unchanged if the
    # index is a tuple or slice, or if the subscript is not being read (writes are handled by the
    # assignment visitor). For plain index reads, looks up the element-type directive defined for
    # the container (default: null value) and returns an expression calling the get-item runtime
    # function with container, key, and an options object carrying the element dtype.
    node = self.generic_visit(node)
    s = node.slice
    if isinstance(s, (gast.Tuple, gast.Slice)):
      return node

    if not isinstance(node.ctx, gast.Load):
      # Index writes are handled at a higher level, one at which the rvalue is
      # also available.
      return node

    dtype = self.get_definition_directive(
        node.value,
        directives.set_element_type,
        'dtype',
        default=templates.replace_as_expression('None'))

    template = """
      ag__.get_item(
          target,
          key,
          opts=ag__.GetItemOpts(element_dtype=dtype))
    """
    return templates.replace_as_expression(
        template, target=node.value, key=s, dtype=dtype)


def transform(node, ctx):
  # [FUNCTION DOC]
  # Entry point of the converter: runs the slice transformer over the AST node using the given
  # transformation context. Returns the transformed node.
  return SliceTransformer(ctx).visit(node)
