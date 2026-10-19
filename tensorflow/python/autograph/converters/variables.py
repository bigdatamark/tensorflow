# [MODULE DOC]
# AutoGraph converter that overloads variable reads. Every name read that existed in the original
# code is wrapped in a call to the AutoGraph load runtime function, which allows tracking
# activity and undefined values. Augmented assignments to plain names get an explicit prior load
# of the target, and deletes of plain names are replaced by rebinding the name to an undefined
# marker value so later use raises a clear error. Exposes a transformer class and a transform
# entry function. No import-time side effects.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Overloads all variable read operations."""

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import templates


class VariableAccessTransformer(converter.Base):
  # [CLASS DOC]
  # AST transformer that wraps variable reads in the load runtime call and rewrites deletes and
  # augmented assignments of plain names. Subclass of the converter base transformer, created
  # with the transformation context by the transform function. Holds no extra state.
  """Rewrites basic symbol reads.

  This transformer rewrites variable reads with a "read" operator which allows
  tracking activity.

  Example:

  For a basic statement:

      a = b + c

  This is translated to:

      a = ld(b) + ld(c)

  Augmented assignment operations also introduce a `ld` operator:

      a += b

  The assignment target also receives an operator to properly represent the
  read:

      a = ld(a)
      a += ld(b)
  """

  def visit_Name(self, node):
    # [FUNCTION DOC]
    # Visitor for names. Names without an original-definitions annotation (i.e. not present in the
    # original source, such as generated ones) are returned unchanged. A name being read is
    # replaced by an expression calling the load runtime function on it; other contexts (stores,
    # deletes) are unchanged. Returns the resulting node.
    # Only the loads which existed in the original code are overloaded.
    if not anno.hasanno(node, anno.Static.ORIG_DEFINITIONS):
      return node
    if isinstance(node.ctx, gast.Load):
      node = templates.replace_as_expression('ag__.ld(var_)', var_=node)
    return node

  def visit_Delete(self, node):
    # [FUNCTION DOC]
    # Visitor for delete statements. Visits children, then for each target that is a plain name
    # (not composites such as indexed or attribute targets) emits an assignment binding the name
    # to an undefined marker object carrying the name, so that later use fails with a clear
    # error. Remaining composite targets are kept in a smaller delete statement appended after.
    # Returns the node itself if nothing was rewritable, otherwise the list of new statements.
    node = self.generic_visit(node)

    rewrite_targets = []
    for tgt in node.targets:
      # Don't rewrite composites like `del a[0]`.
      if isinstance(tgt, gast.Name):
        rewrite_targets.append(tgt)

    if not rewrite_targets:
      return node

    results = []
    for tgt in rewrite_targets:
      template = """
        var_ = ag__.Undefined(var_name)
      """
      results.extend(templates.replace(
          template, var_=tgt, var_name=gast.Constant(tgt.id, kind=None)))
    remaining_targets = [n for n in node.targets if n not in rewrite_targets]
    if remaining_targets:
      results.append(gast.Delete(targets=remaining_targets))

    return results

  def visit_AugAssign(self, node):
    # [FUNCTION DOC]
    # Visitor for augmented assignments. If the target is a plain name, returns two statements: an
    # assignment of the target to the load runtime call on itself (representing the read), then
    # the original augmented assignment. Otherwise visits children generically. Returns the
    # result.
    if isinstance(node.target, gast.Name):
      template = """
        var_ = ag__.ld(var_)
        original
      """
      node = templates.replace(template, var_=node.target, original=node)
    else:
      node = self.generic_visit(node)
    return node


def transform(node, ctx):
  # [FUNCTION DOC]
  # Entry point of the converter: runs the variable access transformer on the AST node with the
  # given transformation context and returns the transformed node.
  return VariableAccessTransformer(ctx).visit(node)
