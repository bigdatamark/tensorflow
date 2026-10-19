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
# [MODULE DOC]
# Defines the qualified-name (QN) data model used by autograph's static analyses, and a node
# transformer that attaches a QN annotation to name, attribute and subscript syntax nodes.
# Exposes: Symbol and Literal (small immutable records), QN (the qualified name itself),
# QnResolver (annotating transformer), resolve (annotate a tree) and from_str (parse text to QN).
# No import-time side effects beyond imports. Not a helper module; other pyct modules
# (activity analysis, etc.) depend on it.
"""Utilities for manipulating qualified names.

A qualified name is a uniform way to refer to simple (e.g. 'foo') and composite
(e.g. 'foo.bar') syntactic symbols.

This is *not* related to the __qualname__ attribute used by inspect, which
refers to scopes.
"""

import collections

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser


class CallerMustSetThis(object):
  # [CLASS DOC]
  # HELPER: sentinel used by QN.ast to mark the expression context of generated nodes.
  # Empty marker class; its class object is placed in the context field of AST nodes built from a
  # QN to signal that the caller must replace it with a real load/store/delete context.
  pass


class Symbol(collections.namedtuple('Symbol', ['name'])):
  # [CLASS DOC]
  # Immutable record with a single field, name, representing a plain symbol. Inherits all
  # behavior from the named tuple; adds nothing.
  """Represents a Python symbol."""


class Literal(collections.namedtuple('Literal', ['value'])):
  # [CLASS DOC]
  # Immutable record with a single field, value, holding a literal constant (such as a number or
  # string) that appears as a subscript index inside a qualified name. Overrides text conversion
  # so strings are shown quoted and other values in their usual text form.
  """Represents a Python numeric literal."""

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the text form: string values wrapped in single quotes, any other value converted
    # with the ordinary text conversion. No side effects.
    if isinstance(self.value, str):
      return "'{}'".format(self.value)
    return str(self.value)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the same text as the text-conversion method. No side effects.
    return str(self)


# TODO(mdan): Use subclasses to remove the has_attr has_subscript booleans.
class QN(object):
  # [CLASS DOC]
  # A qualified name: a simple symbol (like foo), an attribute access on another QN (foo.bar), or
  # a subscript of another QN (foo[bar]). Holds a tuple qn (one element for simple names, two for
  # composite: base QN plus attribute string or subscript QN), a parent link (none for simple
  # names) and booleans recording whether it is an attribute or subscript form. Instances are
  # hashable and comparable (equality uses the tuple and the form flags; ordering uses the text
  # form), so they work as set members and mapping keys. Created directly or through from_str and
  # the QnResolver transformer.
  """Represents a qualified name."""

  def __init__(self, base, attr=None, subscript=None):
    # [FUNCTION DOC]
    # Builds a QN. base is a string or Literal for a simple name, or a QN when attr or subscript
    # is given. attr (optional string) makes an attribute name; subscript (optional QN) makes a
    # subscript name. Returns nothing; sets internal state.
    # Raises ValueError if both attr and subscript are given, if base is not a QN for composite
    # forms, if attr is not a string, or if a simple base is neither string nor Literal.
    # Also asserts (assertion error) that a simple base string contains no dot or brackets.
    if attr is not None and subscript is not None:
      raise ValueError('A QN can only be either an attr or a subscript, not '
                       'both: attr={}, subscript={}.'.format(attr, subscript))
    self._has_attr = False
    self._has_subscript = False

    if attr is not None:
      if not isinstance(base, QN):
        raise ValueError(
            'for attribute QNs, base must be a QN; got instead "%s"' % base)
      if not isinstance(attr, str):
        raise ValueError('attr may only be a string; got instead "%s"' % attr)
      self._parent = base
      # TODO(mdan): Get rid of the tuple - it can only have 1 or 2 elements now.
      self.qn = (base, attr)
      self._has_attr = True

    elif subscript is not None:
      if not isinstance(base, QN):
        raise ValueError('For subscript QNs, base must be a QN.')
      self._parent = base
      self.qn = (base, subscript)
      self._has_subscript = True

    else:
      if not isinstance(base, (str, Literal)):
        # TODO(mdan): Require Symbol instead of string.
        raise ValueError(
            'for simple QNs, base must be a string or a Literal object;'
            ' got instead "%s"' % type(base))
      assert '.' not in base and '[' not in base and ']' not in base
      self._parent = None
      self.qn = (base,)

  def is_symbol(self):
    # [FUNCTION DOC]
    # Returns true if the first element of the name is a string (an identifier) rather than a
    # Literal.
    return isinstance(self.qn[0], str)

  def is_simple(self):
    # [FUNCTION DOC]
    # Returns true if the name has a single element (neither attribute nor subscript).
    return len(self.qn) <= 1

  def is_composite(self):
    # [FUNCTION DOC]
    # Returns true if the name is an attribute or subscript form (more than one element).
    return len(self.qn) > 1

  def has_subscript(self):
    # [FUNCTION DOC]
    # Returns true if this QN is a subscript form.
    return self._has_subscript

  def has_attr(self):
    # [FUNCTION DOC]
    # Returns true if this QN is an attribute form.
    return self._has_attr

  @property
  def attr(self):
    # [FUNCTION DOC]
    # Property returning the attribute string of an attribute-form QN.
    # Raises ValueError if the QN is not an attribute form.
    if not self._has_attr:
      raise ValueError('Cannot get attr of non-attribute "%s".' % self)
    return self.qn[1]

  @property
  def parent(self):
    # [FUNCTION DOC]
    # Property returning the base QN of a composite name (e.g. a.b for a.b.c).
    # Raises ValueError for a simple name, which has no parent.
    if self._parent is None:
      raise ValueError('Cannot get parent of simple name "%s".' % self.qn[0])
    return self._parent

  @property
  def owner_set(self):
    # [FUNCTION DOC]
    # Property returning a new set of all ancestor QNs (parent, grandparent, ...) of this name;
    # empty for simple names. Computed recursively; no side effects.
    """Returns all the symbols (simple or composite) that own this QN.

    In other words, if this symbol was modified, the symbols in the owner set
    may also be affected.

    Examples:
      'a.b[c.d]' has two owners, 'a' and 'a.b'
    """
    owners = set()
    if self.has_attr() or self.has_subscript():
      owners.add(self.parent)
      owners.update(self.parent.owner_set)
    return owners

  @property
  def support_set(self):
    # [FUNCTION DOC]
    # Property returning a new set of simple QNs the name depends on: for an attribute, the
    # support of its parent; for a subscript, the support of the parent plus that of the index
    # QN; for a simple name, the name itself. Note that a literal index is a simple QN and so is
    # included as a member. May raise an attribute error if a subscript index is not a QN.
    """Returns the set of simple symbols that this QN relies on.

    This would be the smallest set of symbols necessary for the QN to
    statically resolve (assuming properties and index ranges are verified
    at runtime).

    Examples:
      'a.b' has only one support symbol, 'a'
      'a[i]' has two support symbols, 'a' and 'i'
    """
    # TODO(mdan): This might be the set of Name nodes in the AST. Track those?
    roots = set()
    if self.has_attr():
      roots.update(self.parent.support_set)
    elif self.has_subscript():
      roots.update(self.parent.support_set)
      roots.update(self.qn[1].support_set)
    else:
      roots.add(self)
    return roots

  def __hash__(self):
    # [FUNCTION DOC]
    # Returns a hash of the element tuple combined with the attribute and subscript flags,
    # consistent with equality.
    return hash(self.qn + (self._has_attr, self._has_subscript))

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Returns true if other is a QN with equal element tuple and the same attribute and
    # subscript flags; false otherwise (including for non-QN values).
    return (isinstance(other, QN) and self.qn == other.qn and
            self.has_subscript() == other.has_subscript() and
            self.has_attr() == other.has_attr())

  def __lt__(self, other):
    # [FUNCTION DOC]
    # Orders QNs by comparing their text forms (string comparison); gives a stable sort order.
    return str(self) < str(other)

  def __gt__(self, other):
    # [FUNCTION DOC]
    # Greater-than counterpart of the less-than method, comparing text forms.
    return str(self) > str(other)

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the source-like text: root[index] for subscripts, dotted names for attributes, or
    # just the root for simple names.
    root = self.qn[0]
    if self.has_subscript():
      return '{}[{}]'.format(root, self.qn[1])
    if self.has_attr():
      return '.'.join(map(str, self.qn))
    else:
      return str(root)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the same text as the text-conversion method.
    return str(self)

  def ssf(self):
    # [FUNCTION DOC]
    # Returns the "simple symbol form": a valid-identifier-like string joining the elements with
    # an underscore for attributes, or a "_sub_" marker for subscripts, applied recursively to
    # nested QNs. Pure; no side effects.
    """Simple symbol form."""
    ssfs = [n.ssf() if isinstance(n, QN) else n for n in self.qn]
    ssf_string = ''
    for i in range(0, len(self.qn) - 1):
      if self.has_subscript():
        delimiter = '_sub_'
      else:
        delimiter = '_'
      ssf_string += ssfs[i] + delimiter
    return ssf_string + ssfs[-1]

  def ast(self):
    # [FUNCTION DOC]
    # Builds and returns a gast expression node equivalent to this name: a subscript node, an
    # attribute node, a name node, or a constant node for literal bases. Nested parts are built
    # recursively. The context field of name, attribute and subscript nodes is set to the
    # CallerMustSetThis sentinel; callers must replace it with a real context.
    # KNOWN ISSUE: for subscripts the index is converted via the index QN's own conversion, so a
    # non-QN index would fail; the final branch is an unreachable assertion guarded by the
    # constructor.
    """AST representation."""
    # The caller must adjust the context appropriately.
    if self.has_subscript():
      return gast.Subscript(
          value=self.parent.ast(),
          slice=self.qn[-1].ast(),
          ctx=CallerMustSetThis)
    if self.has_attr():
      return gast.Attribute(
          value=self.parent.ast(), attr=self.qn[-1], ctx=CallerMustSetThis)

    base = self.qn[0]
    if isinstance(base, str):
      return gast.Name(
          base, ctx=CallerMustSetThis, annotation=None, type_comment=None)
    elif isinstance(base, Literal):
      return gast.Constant(base.value, kind=None)
    else:
      assert False, ('the constructor should prevent types other than '
                     'str and Literal')


class QnResolver(gast.NodeTransformer):
  # [CLASS DOC]
  # Tree transformer that attaches a QN annotation (under the basic QN annotation key) to name,
  # attribute and subscript nodes, bottom-up so that child annotations exist when parents are
  # processed. Stateless. Modifies nodes in place and returns them. Used through resolve.
  """Annotates nodes with QN information.

  Note: Not using NodeAnnos to avoid circular dependencies.
  """

  def visit_Name(self, node):
    # [FUNCTION DOC]
    # Visits children, then annotates the name node with a simple QN of its identifier.
    # Returns the same node.
    node = self.generic_visit(node)
    anno.setanno(node, anno.Basic.QN, QN(node.id))
    return node

  def visit_Attribute(self, node):
    # [FUNCTION DOC]
    # Visits children, then, if the value expression carries a QN annotation, annotates this
    # node with an attribute QN built on it. Otherwise leaves it unannotated. Returns the node.
    node = self.generic_visit(node)
    if anno.hasanno(node.value, anno.Basic.QN):
      anno.setanno(node, anno.Basic.QN,
                   QN(anno.getanno(node.value, anno.Basic.QN), attr=node.attr))
    return node

  def visit_Subscript(self, node):
    # [FUNCTION DOC]
    # Visits children, then annotates the node with a subscript QN when possible. Tuple and slice
    # indices are silently skipped (node returned unannotated). A constant index (other than the
    # ellipsis) becomes a literal QN; otherwise the index's own QN annotation is used, and if
    # absent the node is left unannotated. The annotation is only set if the indexed value has a
    # QN annotation. Returns the node.
    # TODO(mdan): This may no longer apply if we overload getitem.
    node = self.generic_visit(node)
    s = node.slice
    if isinstance(s, (gast.Tuple, gast.Slice)):
      # TODO(mdan): Support range and multi-dimensional indices.
      # Continuing silently because some demos use these.
      return node
    if isinstance(s, gast.Constant) and s.value != Ellipsis:
      subscript = QN(Literal(s.value))
    else:
      # The index may be an expression, case in which a name doesn't make sense.
      if anno.hasanno(s, anno.Basic.QN):
        subscript = anno.getanno(s, anno.Basic.QN)
      else:
        return node
    if anno.hasanno(node.value, anno.Basic.QN):
      anno.setanno(node, anno.Basic.QN,
                   QN(anno.getanno(node.value, anno.Basic.QN),
                      subscript=subscript))
    return node


def resolve(node):
  # [FUNCTION DOC]
  # Annotates the given syntax tree in place with qualified names using a QnResolver and returns
  # the resulting root node.
  return QnResolver().visit(node)


def from_str(qn_str):
  # [FUNCTION DOC]
  # Parses a string holding a single expression (such as a dotted or subscripted name), resolves
  # qualified names on it and returns the QN annotation of the root. Raises whatever the parser
  # raises on invalid text, or a missing-annotation error if the expression has no QN.
  node = parser.parse_expression(qn_str)
  node = resolve(node)
  return anno.getanno(node, anno.Basic.QN)
