# [MODULE DOC]
# Conversion of Python ASTs (gast) to A-normal form (ANF), where intermediate values are
# explicitly named by fresh temporary variables. Exposes the public transform function, a
# configurable AnfTransformer, the edge-pattern class used to describe which parent/field/child
# edges to rewrite, and the constants REPLACE, LEAVE and ANY used in configurations. Also contains
# a simple fresh-name generator and private triviality helpers. Import-time side effects: only
# defines module-level constants.
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
"""Conversion to A-normal form.

The general idea of A-normal form is that every intermediate value is
explicitly named with a variable.  For more, see
https://en.wikipedia.org/wiki/A-normal_form.

The specific converters used here are based on Python AST semantics as
documented at https://greentreesnakes.readthedocs.io/en/latest/.
"""

import collections

import gast

from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer


# TODO(mdan): Replace with naming.Namer.
class DummyGensym:
  # [CLASS DOC]
  # HELPER: fresh-name generator for the ANF transformer.
  # Keeps a counter and produces unique temporary names by appending an increasing number
  # (starting just above one thousand) to a stem. Does not check for collisions with existing
  # symbols.
  """A dumb gensym that suffixes a stem by sequential numbers from 1000."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Takes no arguments; sets the counter to zero. Returns nothing.
    # A proper implementation needs to account for:
    #   * ctx.info.namespace
    #   * all the symbols defined in the AST
    #   * the symbols generated so far
    self._idx = 0

  def new_name(self, stem='tmp'):
    # [FUNCTION DOC]
    # HELPER: name source for the ANF transformer.
    # Increments the counter and returns the stem (default "tmp"), an underscore and the
    # number one thousand plus the counter, as a string. Mutates the counter.
    self._idx += 1
    return stem + '_' + str(1000 + self._idx)


REPLACE = lambda _1, _2, _3: True
LEAVE = lambda _1, _2, _3: False
ANY = object()


class ASTEdgePattern(collections.namedtuple(
    'ASTEdgePattern', ['parent', 'field', 'child'])):
  # [CLASS DOC]
  # Immutable record (named tuple with three slots: parent, field, child) describing a kind of AST
  # edge. The parent and child slots hold node types and the field slot holds a field name; any
  # slot may be the wildcard ANY. Used in ANF configurations to select which edges to rewrite.
  """A pattern defining a type of AST edge.

  This consists of three components:
  - The type of the parent node, checked with isinstance,
  - The name of the field, checked with string equality, and
  - The type of the child node, also checked with isinstance.
  If all three match, the whole pattern is considered to match.

  In all three slots, the special value `anf.ANY` is treated as "match
  anything".  The internal nodes are produced from the `gast` library rather
  than the standard `ast` module, which may affect `isinstance` checks.
  """
  __slots__ = ()

  def matches(self, parent, field, child):
    # [FUNCTION DOC]
    # Checks one concrete edge against the pattern. Inputs: parent node, field name (string) and
    # child node. The parent and child match if the slot is ANY or the node is an instance of the
    # slot's type; the field matches if the slot is ANY or the names are equal. Returns true only
    # if all three match.
    """Computes whether this pattern matches the given edge."""
    if self.parent is ANY or isinstance(parent, self.parent):
      pass  # OK
    else:
      return False
    if self.field is ANY or field == self.field:
      pass  # OK
    else:
      return False
    return self.child is ANY or isinstance(child, self.child)


class AnfTransformer(transformer.Base):
  # [CLASS DOC]
  # AST transformer that rewrites a tree into A-normal form. Holds a list of override rules
  # (pattern, decision) deciding which edges get a fresh variable, a fresh-name generator, and a
  # list of pending assignment statements accumulated while walking expressions; pending
  # statements are flushed in front of the enclosing statement. Created with a transformer
  # context and an optional configuration; used by calling visit on a tree (see transform).
  # Unsupported constructs (comprehensions, nontrivial lambdas, conditional expressions, etc.)
  # raise value errors.
  """Performs the conversion to A-normal form (ANF)."""

  # The algorithm is a postorder recursive tree walk.  Any given node A may, in
  # general, require creation of a series B of Assign statements, which compute
  # and explicitly name the intermediate values needed to compute the value of
  # A.  If A was already a statement, it can be replaced with the sequence B +
  # [A].  If A was an expression, B needs to be propagated up the tree until a
  # statement is encountered.  Since the `ast.NodeTransformer` framework makes
  # no provision for subtraversals returning side information, this class
  # accumulates the sequence B in an instance variable.

  # The only other subtlety is that some Python statements (like `if`) have both
  # expression fields (`test`) and statement list fields (`body` and `orelse`).
  # Any additional assignments needed to name all the intermediate values in the
  # `test` can be prepended to the `if` node, but assignments produced by
  # processing the `body` and the `orelse` need to be kept together with them,
  # and not accidentally lifted out of the `if`.

  def __init__(self, ctx, config):
    # [FUNCTION DOC]
    # Constructor. Inputs: ctx (the transformer context) and config (list of rules pairing an edge
    # pattern with replace, leave, or a callable; or nothing for the default). The default leaves
    # literal constants and names alone and replaces all other expressions. Creates the name
    # generator and an empty pending-statement list. Asserts (fails) if neither supported gast
    # version flag holds.
    """Creates an ANF transformer.

    Args:
      ctx: transformer.Context
      config: Configuration
    """
    super(AnfTransformer, self).__init__(ctx)
    if config is None:
      # These could be pulled out, but are generally considered to already be in
      # A-normal form.  Thus they are left in by default, but could be pulled
      # out if the configuration calls for it.
      if gast_util.GAST2:
        literal_node_types = (
            gast.Num, gast.Str, gast.Bytes, gast.NameConstant,
            gast.Name  # Name is here to cover True, False, and None in Python 2
        )
      elif gast_util.GAST3:
        literal_node_types = (
            gast.Constant,
            gast.Name  # Name is here to cover True, False, and None in Python 2
        )
      else:
        assert False

      self._overrides = [
          (ASTEdgePattern(ANY, ANY, literal_node_types), LEAVE),
          (ASTEdgePattern(ANY, ANY, gast.expr), REPLACE)]
    else:
      self._overrides = config
    self._gensym = DummyGensym()
    self._pending_statements = []

  def _consume_pending_statements(self):
    # [FUNCTION DOC]
    # HELPER: pending-statement bookkeeping for the ANF visitors.
    # Returns the accumulated list of pending assignments and resets the stored list to empty.
    ans = self._pending_statements
    self._pending_statements = []
    return ans

  def _add_pending_statement(self, stmt):
    # [FUNCTION DOC]
    # HELPER: pending-statement bookkeeping for the ANF visitors.
    # Appends one statement to the pending list. Returns nothing.
    self._pending_statements.append(stmt)

  def _match(self, pattern, parent, field, child):
    # [FUNCTION DOC]
    # HELPER: rule matching for the ANF override list.
    # Returns true if the pattern is the wildcard, otherwise the result of asking the pattern
    # whether it matches the given parent, field and child.
    if pattern is ANY:
      return True
    else:
      return pattern.matches(parent, field, child)

  def _should_transform(self, parent, field, child):
    # [FUNCTION DOC]
    # HELPER: decides whether an edge's child should be replaced by a variable.
    # Walks the override rules in order; the first rule whose pattern matches decides, by calling
    # its decision (replace, leave, or a callable) with the parent, field and child, and its
    # boolean result is returned. If no rule matches, returns false.
    for pat, result in self._overrides:
      if self._match(pat, parent, field, child):
        return result(parent, field, child)
    # Fell off the end of the pattern list: do not transform
    return False

  def _do_transform_node(self, node):
    # [FUNCTION DOC]
    # HELPER: performs the actual extraction of an expression into a temporary variable.
    # Generates a fresh name, queues a pending assignment of the expression to that name, and
    # returns a name-reference node for the new variable to replace the expression.
    temp_name = self._gensym.new_name()
    temp_assign = templates.replace(
        'temp_name = expr', temp_name=temp_name, expr=node)[0]
    self._add_pending_statement(temp_assign)
    answer = templates.replace('temp_name', temp_name=temp_name)[0]
    return answer

  def _ensure_node_in_anf(self, parent, field, node):
    # [FUNCTION DOC]
    # HELPER: core of the per-edge ANF decision.
    # Inputs: parent node, field name, and the child node (which may be nothing, a list, or a
    # node). Nothing and trivial nodes are returned unchanged; lists are processed element by
    # element; keyword nodes have their value processed; starred, with-item and slice nodes have
    # their own fields processed using the inherited parent and field; otherwise, if the rules say
    # to transform, the node is replaced by a fresh variable reference (queueing an assignment),
    # else returned unchanged. Returns the node or its replacement.
    """Puts `node` in A-normal form, by replacing it with a variable if needed.

    The exact definition of A-normal form is given by the configuration.  The
    parent and the incoming field name are only needed because the configuration
    may be context-dependent.

    Args:
      parent: An AST node, the parent of `node`.
      field: The field name under which `node` is the child of `parent`.
      node: An AST node, potentially to be replaced with a variable reference.

    Returns:
      node: An AST node; the argument if transformation was not necessary,
        or the new variable reference if it was.
    """
    if node is None:
      return node
    if _is_trivial(node):
      return node
    if isinstance(node, list):
      # If something's field was actually a list, e.g., variadic arguments.
      return [self._ensure_node_in_anf(parent, field, n) for n in node]
    if isinstance(node, gast.keyword):
      node.value = self._ensure_node_in_anf(parent, field, node.value)
      return node
    if isinstance(node, (gast.Starred, gast.withitem, gast.slice)):
      # These nodes aren't really extractable in their own right, but their
      # subnodes might be.  Propagate the parent and field name to the child
      # nodes, instead of querying the configuration for children of, e.g.,
      # gast.Starred.
      return self._ensure_fields_in_anf(node, parent, field)
    if self._should_transform(parent, field, node):
      return self._do_transform_node(node)
    else:
      return node

  def _ensure_fields_in_anf(self, node, parent=None, super_field=None):
    # [FUNCTION DOC]
    # HELPER: applies the per-edge ANF decision to every field of a node.
    # Inputs: node, and optional parent and field-name overrides (used when the node is a
    # wrapper such as starred; by default the node itself and each field's own name are used).
    # Skips fields whose names begin with two underscores. Replaces each field value in place
    # with the processed value. Returns the same node.
    for field in node._fields:
      if field.startswith('__'):
        continue
      parent_supplied = node if parent is None else parent
      field_supplied = field if super_field is None else super_field
      setattr(node, field, self._ensure_node_in_anf(
          parent_supplied, field_supplied, getattr(node, field)))
    return node

  def _visit_strict_statement(self, node, children_ok_to_transform=True):
    # [FUNCTION DOC]
    # HELPER: shared visitor for statements whose expression fields may be named.
    # Inputs: the statement node and a flag (default true) saying whether its fields may be
    # replaced by variables. Asserts no statements are pending, visits children, optionally
    # puts the fields in ANF, then returns a list: all pending assignments followed by the node.
    assert not self._pending_statements
    node = self.generic_visit(node)
    if children_ok_to_transform:
      self._ensure_fields_in_anf(node)
    results = self._consume_pending_statements()
    results.append(node)
    return results

  def _visit_trivial_only_statement(self, node, msg):
    # [FUNCTION DOC]
    # HELPER: shared visitor for statements that are only supported when already trivial.
    # Inputs: node and an error message. Visits children and puts fields in ANF; if that would
    # have required new pending statements, raises a value error with the message. Otherwise
    # returns the node.
    assert not self._pending_statements
    node = self.generic_visit(node)
    self._ensure_fields_in_anf(node)
    if self._pending_statements:
      raise ValueError(msg)
    else:
      return node

  def _visit_strict_expression(self, node):
    # [FUNCTION DOC]
    # HELPER: shared visitor for expressions. Visits children, puts fields in ANF (queueing
    # pending assignments), and returns the node.
    node = self.generic_visit(node)
    self._ensure_fields_in_anf(node)
    return node

  def _visit_trivial_only_expression(self, node, msg):
    # [FUNCTION DOC]
    # HELPER: shared visitor for expressions only supported when already trivial.
    # Inputs: node and an error message. Remembers the number of pending statements, visits
    # children and puts fields in ANF; raises a value error with the message if the count of
    # pending statements changed, else returns the node.
    k = len(self._pending_statements)
    node = self.generic_visit(node)
    self._ensure_fields_in_anf(node)
    # This check relies on there being no opportunities to consume pending
    # statements while traversing children of an expression.
    if len(self._pending_statements) != k:
      raise ValueError(msg)
    else:
      return node

  # Note on code order: These are listed in the same order as the grammar
  # elements on https://github.com/serge-sans-paille/gast

  # FunctionDef, AsyncFunctionDef, and ClassDef should be correct by default.

  def visit_Return(self, node):
    # [FUNCTION DOC]
    # Visitor for return statements: names intermediate values in the returned expression.
    # Returns a list of pending assignments followed by the node.
    return self._visit_strict_statement(node)

  def visit_Delete(self, node):
    # [FUNCTION DOC]
    # Visitor for delete statements. Children are visited but the deleted targets are never
    # replaced with variables. Returns a list of pending assignments followed by the node.
    return self._visit_strict_statement(node, children_ok_to_transform=False)

  def visit_Assign(self, node):
    # [FUNCTION DOC]
    # Visitor for assignments. Children are visited (nested expressions may queue assignments)
    # but the assignment's own targets and value are not replaced. Returns a list of pending
    # assignments followed by the node.
    return self._visit_strict_statement(node, children_ok_to_transform=False)

  def visit_AugAssign(self, node):
    # [FUNCTION DOC]
    # Visitor for augmented assignments; same policy as plain assignment: the target and value
    # themselves are not replaced. Returns a list of pending assignments followed by the node.
    return self._visit_strict_statement(node, children_ok_to_transform=False)

  def visit_Print(self, node):
    # [FUNCTION DOC]
    # Visitor for the legacy print statement node (Python 2 style): names intermediate values in
    # its arguments. Returns a list of pending assignments followed by the node.
    return self._visit_strict_statement(node)

  def visit_For(self, node):
    # [FUNCTION DOC]
    # Visitor for for-loops. Visits the iterable first and puts it in ANF so that any assignments
    # created go before the loop (not inside the body), then visits the whole loop. Returns a
    # list of those hoisted assignments followed by the loop node. Asserts no statements pending.
    assert not self._pending_statements
    # It's important to visit node.iter first, because any statements created
    # thereby need to live outside the body.
    self.visit(node.iter)
    node.iter = self._ensure_node_in_anf(node, 'iter', node.iter)
    iter_stmts = self._consume_pending_statements()
    # This generic_visit will revisit node.iter, but that is correct because by
    # this point the node.iter link has been checked.  It may be somewhat
    # expensive if the configuration didn't call for transforming node.iter, as
    # then it may be large and will be uselessly transformed again.  This
    # behavior is what causes the documented effect that configuration callables
    # may be invoked more than once of the same links; if the code is rewritten
    # not to do that (anywhere), the docstring of `transform` should be updated.
    node = self.generic_visit(node)
    assert not self._pending_statements
    iter_stmts.append(node)
    return iter_stmts

  def visit_AsyncFor(self, node):
    # [FUNCTION DOC]
    # Visitor for async for-loops. Only supported when nothing needs to be extracted; otherwise
    # raises a value error saying nontrivial async for nodes are unsupported. Returns the node.
    msg = ('Nontrivial AsyncFor nodes not supported yet '
           '(need to think through the semantics).')
    return self._visit_trivial_only_statement(node, msg)

  def visit_While(self, node):
    # [FUNCTION DOC]
    # Visitor for while-loops. Visits and puts the test in ANF; if that would require extracting
    # statements (they would be evaluated only once), raises a value error. Otherwise visits the
    # loop generically and returns the node.
    assert not self._pending_statements
    self.visit(node.test)
    node.test = self._ensure_node_in_anf(node, 'test', node.test)
    if self._pending_statements:
      msg = ('While with nontrivial test not supported yet '
             '(need to avoid precomputing the test).')
      raise ValueError(msg)
    # If traversing node.test yielded no statements extracted, the generic visit
    # will do the right thing.
    return self.generic_visit(node)

  def visit_If(self, node):
    # [FUNCTION DOC]
    # Visitor for if statements. Visits the test first and puts it in ANF so created assignments
    # precede the if, then visits the rest (branches keep their own assignments). Returns a list
    # of the hoisted assignments followed by the if node. Asserts no statements pending.
    assert not self._pending_statements
    # It's important to visit node.test first, because any statements created
    # thereby need to live outside the body.
    self.visit(node.test)
    node.test = self._ensure_node_in_anf(node, 'test', node.test)
    condition_stmts = self._consume_pending_statements()
    # This generic_visit will revisit node.test, but that is correct because by
    # this point the node.test link has been checked.  It may be somewhat
    # expensive if the configuration didn't call for transforming node.test, as
    # then it may be large and will be uselessly transformed again.  This
    # happens in several places.
    node = self.generic_visit(node)
    assert not self._pending_statements
    condition_stmts.append(node)
    return condition_stmts

  def visit_With(self, node):
    # [FUNCTION DOC]
    # Visitor for with statements. Visits the context items and puts them in ANF so created
    # assignments precede the with, then visits the rest. Returns a list of the hoisted
    # assignments followed by the node. Asserts no statements pending.
    assert not self._pending_statements
    # It's important to visit node.items first, because any statements created
    # thereby need to live outside the body.
    for item in node.items:
      self.visit(item)
    node.items = [self._ensure_node_in_anf(node, 'items', n)
                  for n in node.items]
    contexts_stmts = self._consume_pending_statements()
    # This generic_visit will revisit node.items, but that is correct because by
    # this point the node.items link has been checked.  It may be somewhat
    # expensive if the configuration didn't call for transforming node.items, as
    # then it may be large and will be uselessly transformed again.  This
    # happens in several places.
    node = self.generic_visit(node)
    assert not self._pending_statements
    contexts_stmts.append(node)
    return contexts_stmts

  def visit_AsyncWith(self, node):
    # [FUNCTION DOC]
    # Visitor for async with statements; only supported when nothing needs to be extracted,
    # otherwise raises a value error. Returns the node.
    msg = ('Nontrivial AsyncWith nodes not supported yet '
           '(need to think through the semantics).')
    return self._visit_trivial_only_statement(node, msg)

  def visit_Raise(self, node):
    # [FUNCTION DOC]
    # Visitor for raise statements: names intermediate values in the raised expressions. Returns
    # a list of pending assignments followed by the node.
    return self._visit_strict_statement(node)

  # Try should be correct by default.

  def visit_Assert(self, node):
    # [FUNCTION DOC]
    # Visitor for assert statements; only supported when nothing needs to be extracted (to keep
    # lazy evaluation semantics), otherwise raises a value error. Returns the node.
    msg = ('Nontrivial Assert nodes not supported yet '
           '(need to avoid computing the test when assertions are off, and '
           'avoid computing the irritant when the assertion does not fire).')
    return self._visit_trivial_only_statement(node, msg)

  # Import and ImportFrom should be correct by default.

  def visit_Exec(self, node):
    # [FUNCTION DOC]
    # Visitor for the legacy exec statement node (Python 2 style): names intermediate values.
    # Returns a list of pending assignments followed by the node.
    return self._visit_strict_statement(node)

  # Global and Nonlocal should be correct by default.

  def visit_Expr(self, node):
    # [FUNCTION DOC]
    # Visitor for expression statements. Children are visited but the top-level expression is not
    # itself replaced. Returns a list of pending assignments followed by the node.
    return self._visit_strict_statement(node, children_ok_to_transform=False)

  # Pass, Break, and Continue should be correct by default.

  def visit_BoolOp(self, node):
    # [FUNCTION DOC]
    # Visitor for boolean and/or expressions; only supported when nothing needs to be extracted
    # (to preserve short-circuiting), otherwise raises a value error. Returns the node.
    msg = ('Nontrivial BoolOp nodes not supported yet '
           '(need to preserve short-circuiting semantics).')
    return self._visit_trivial_only_expression(node, msg)

  def visit_BinOp(self, node):
    # [FUNCTION DOC]
    # Visitor for binary operations: visits children and names operands per the configuration.
    # Returns the node.
    return self._visit_strict_expression(node)

  def visit_UnaryOp(self, node):
    # [FUNCTION DOC]
    # Visitor for unary operations: visits children and names the operand per the configuration.
    # Returns the node.
    return self._visit_strict_expression(node)

  def visit_Lambda(self, node):
    # [FUNCTION DOC]
    # Visitor for lambda expressions; statements cannot be inserted into lambda bodies, so it
    # raises a value error if any extraction would be needed. Returns the node otherwise.
    msg = ('Nontrivial Lambda nodes not supported '
           '(cannot insert statements into lambda bodies).')
    return self._visit_trivial_only_expression(node, msg)

  def visit_IfExp(self, node):
    # [FUNCTION DOC]
    # Visitor for conditional expressions; only supported when nothing needs to be extracted
    # (branches must stay lazy), otherwise raises a value error. Returns the node.
    msg = ('Nontrivial IfExp nodes not supported yet '
           '(need to convert to If statement, to evaluate branches lazily '
           'and insert statements into them).')
    return self._visit_trivial_only_expression(node, msg)

  def visit_Dict(self, node):
    # [FUNCTION DOC]
    # Visitor for dictionary displays: visits children and names keys and values per the
    # configuration. Returns the node.
    return self._visit_strict_expression(node)

  def visit_Set(self, node):
    # [FUNCTION DOC]
    # Visitor for set displays: visits children and names elements per the configuration.
    # Returns the node.
    return self._visit_strict_expression(node)

  def visit_ListComp(self, node):
    # [FUNCTION DOC]
    # Visitor for list comprehensions; always raises a value error (not supported).
    msg = ('ListComp nodes not supported '
           '(need to convert to a form that tolerates '
           'assignment statements in clause bodies).')
    raise ValueError(msg)

  def visit_SetComp(self, node):
    # [FUNCTION DOC]
    # Visitor for set comprehensions; always raises a value error (not supported).
    msg = ('SetComp nodes not supported '
           '(need to convert to a form that tolerates '
           'assignment statements in clause bodies).')
    raise ValueError(msg)

  def visit_DictComp(self, node):
    # [FUNCTION DOC]
    # Visitor for dictionary comprehensions; always raises a value error (not supported).
    msg = ('DictComp nodes not supported '
           '(need to convert to a form that tolerates '
           'assignment statements in clause bodies).')
    raise ValueError(msg)

  def visit_GeneratorExp(self, node):
    # [FUNCTION DOC]
    # Visitor for generator expressions; always raises a value error (not supported).
    msg = ('GeneratorExp nodes not supported '
           '(need to convert to a form that tolerates '
           'assignment statements in clause bodies).')
    raise ValueError(msg)

  def visit_Await(self, node):
    # [FUNCTION DOC]
    # Visitor for await expressions; only supported when nothing needs to be extracted, otherwise
    # raises a value error. Returns the node.
    msg = ('Nontrivial Await nodes not supported yet '
           '(need to think through the semantics).')
    return self._visit_trivial_only_expression(node, msg)

  def visit_Yield(self, node):
    # [FUNCTION DOC]
    # Visitor for yield expressions: visits children and names the yielded value per the
    # configuration. Returns the node.
    return self._visit_strict_expression(node)

  def visit_YieldFrom(self, node):
    # [FUNCTION DOC]
    # Visitor for yield-from expressions; only supported when nothing needs to be extracted,
    # otherwise raises a value error. Returns the node.
    msg = ('Nontrivial YieldFrom nodes not supported yet '
           '(need to unit-test them in Python 2).')
    return self._visit_trivial_only_expression(node, msg)

  def visit_Compare(self, node):
    # [FUNCTION DOC]
    # Visitor for comparisons. Chained comparisons (more than one operator) raise a value error
    # to preserve short-circuiting; otherwise visits children and names operands per the
    # configuration. Returns the node.
    if len(node.ops) > 1:
      msg = ('Multi-ary compare nodes not supported yet '
             '(need to preserve short-circuiting semantics).')
      raise ValueError(msg)
    return self._visit_strict_expression(node)

  def visit_Call(self, node):
    # [FUNCTION DOC]
    # Visitor for calls: visits children and names the callee and arguments per the
    # configuration. Returns the node.
    return self._visit_strict_expression(node)

  def visit_Repr(self, node):
    # [FUNCTION DOC]
    # Visitor for the legacy backquote repr node (Python 2 style); only supported when nothing
    # needs to be extracted, otherwise raises a value error. Returns the node.
    msg = ('Nontrivial Repr nodes not supported yet '
           '(need to research their syntax and semantics).')
    return self._visit_trivial_only_expression(node, msg)

  def visit_FormattedValue(self, node):
    # [FUNCTION DOC]
    # Visitor for formatted-string value fields; only supported when nothing needs to be
    # extracted, otherwise raises a value error. Returns the node.
    msg = ('Nontrivial FormattedValue nodes not supported yet '
           '(need to unit-test them in Python 2).')
    return self._visit_trivial_only_expression(node, msg)

  def visit_JoinedStr(self, node):
    # [FUNCTION DOC]
    # Visitor for formatted string literals; only supported when nothing needs to be extracted,
    # otherwise raises a value error. Returns the node.
    msg = ('Nontrivial JoinedStr nodes not supported yet '
           '(need to unit-test them in Python 2).')
    return self._visit_trivial_only_expression(node, msg)

  def visit_Attribute(self, node):
    # [FUNCTION DOC]
    # Visitor for attribute access: visits children and names the base value per the
    # configuration. Returns the node.
    return self._visit_strict_expression(node)

  def visit_Subscript(self, node):
    # [FUNCTION DOC]
    # Visitor for subscripts: visits children and names the base and index per the
    # configuration. Returns the node.
    return self._visit_strict_expression(node)

  # Starred and Name are correct by default, because the right thing to do is to
  # just recur.

  def visit_List(self, node):
    # [FUNCTION DOC]
    # Visitor for list displays. Visits children; unless the list is an assignment target
    # (store context), names its elements per the configuration. Returns the node.
    node = self.generic_visit(node)
    if not isinstance(node.ctx, gast.Store):
      self._ensure_fields_in_anf(node)
    return node

  def visit_Tuple(self, node):
    # [FUNCTION DOC]
    # Visitor for tuple displays. Visits children; unless the tuple is an assignment target
    # (store context), names its elements per the configuration. Returns the node.
    node = self.generic_visit(node)
    if not isinstance(node.ctx, gast.Store):
      self._ensure_fields_in_anf(node)
    return node


def _is_py2_name_constant(node):
  # [FUNCTION DOC]
  # HELPER: triviality check for the ANF conversion.
  # Returns true if the node is a name node whose identifier is one of the constants true,
  # false or none (how Python 2 represented them).
  return isinstance(node, gast.Name) and node.id in ['True', 'False', 'None']


def _is_trivial(node):
  # [FUNCTION DOC]
  # HELPER: tells the ANF transformer which nodes never need a temporary variable.
  # Returns true for names (except the py2-style constants), booleans, strings, operator nodes,
  # expression contexts and the ellipsis; false otherwise.
  """Returns whether to consider the given node 'trivial'.

  The definition of 'trivial' is a node that can't meaningfully be pulled out
  into its own assignment statement.

  This is surprisingly difficult to do robustly across versions of Python and
  gast, as the parsing of constants has changed, if I may, constantly.

  Args:
    node: An AST node to check for triviality

  Returns:
    trivial: A Python `bool` indicating whether the node is trivial.
  """
  trivial_node_types = (
      # Variable names
      gast.Name,
      # Non-nodes that show up as AST fields
      bool,
      str,
      # Binary operators
      gast.Add,
      gast.Sub,
      gast.Mult,
      gast.Div,
      gast.Mod,
      gast.Pow,
      gast.LShift,
      gast.RShift,
      gast.BitOr,
      gast.BitXor,
      gast.BitAnd,
      gast.FloorDiv,
      # Unary operators
      gast.Invert,
      gast.Not,
      gast.UAdd,
      gast.USub,
      # Comparison operators
      gast.Eq,
      gast.NotEq,
      gast.Lt,
      gast.LtE,
      gast.Gt,
      gast.GtE,
      gast.Is,
      gast.IsNot,
      gast.In,
      gast.NotIn,
      # Other leaf nodes that don't make sense standalone.
      gast.expr_context,
  )
  if isinstance(node, trivial_node_types) and not _is_py2_name_constant(node):
    return True
  if gast_util.is_ellipsis(node):
    return True

  return False


def transform(node, ctx, config=None):
  # [FUNCTION DOC]
  # Public entry point. Converts the AST node to A-normal form. Inputs: node (tree to convert),
  # ctx (transformer context), config (optional list of rules; default replaces all expressions
  # except literals). Builds an ANF transformer and visits the node. Returns the transformed
  # node (or a list of nodes when the input is a statement). Raises a value error for
  # unsupported constructs.
  # KNOWN ISSUE: the docstring calls ctx an EntityInfo, but the transformer base expects a
  # transformer context.
  """Converts the given node to A-normal form (ANF).

  The general idea of A-normal form: https://en.wikipedia.org/wiki/A-normal_form

  The specific converters used here are based on Python AST semantics as
  documented at https://greentreesnakes.readthedocs.io/en/latest/.

  What exactly should be considered A-normal form for any given programming
  language is not completely obvious.  The transformation defined here is
  therefore configurable as to which syntax to replace with a fresh variable and
  which to leave be.  The configuration is intentionally flexible enough to
  define very precise variable insertion transformations, should that be
  desired.

  The configuration is a list of syntax rules, each of which is a 2-tuple:
  - An `ASTEdgePattern` (which see) defining a type of AST edge, and
  - Whether to transform children of such edges.
  The special object `anf.ANY` may be used as a pattern that matches all edges.

  Each replacement directive is one of three possible things:
  - The object `anf.REPLACE`, meaning "Replace this child node with a variable",
  - The object `anf.LEAVE`, meaning "Do not replace this child node with a
    variable", or
  - A Python callable.  If a callable, it is called with the parent node, the
    field name, and the child node, and must compute a boolean indicating
    whether to transform the child node or not.  The callable is free to use
    whatever context information it chooses.  The callable may be invoked more
    than once on the same link, and must produce the same answer each time.

  The syntax rules are tested in order, and the first match governs.  If no rule
  matches, the node is not transformed.

  The above rules notwithstanding,
  - Variable references are never replaced with (fresh) variables, as that would
    accomplish nothing.
  - The left-hand children of Assign and AugAssign nodes, and the children of
    Del nodes, are never replaced with variables, as that would break their
    semantics.
  - The right-hand children of Assign nodes are never replaced with variables,
    as the original assignment would still have to be present in the result
    to define the new variable.  (That is, there's no point in transforming
    `x = sin(y)` into `tmp = sin(y); x = tmp`.)
  - The right-hand children of AugAssign nodes are never replaced with variables
    either, but only because the difference from Assign was considered a
    potential source of confusion (and it would have been slightly awkward in
    the code to treat the RHS differently than the LHS).
  - Various special-purpose AST nodes are not exposed to the configuration, lest
    the transform produce invalid syntax like, e.g., `tmp = +; x = 1 tmp 2`.

  For example, the configuration
  ```python
  [(anf.ASTEdgePattern(anf.ANY, anf.ANY, gast.expr), anf.REPLACE)]
  ```
  gives explicit fresh names to all expressions regardless of context (except as
  outlined above), whereas
  ```python
  [(anf.ASTEdgePattern(gast.If, "test", anf.ANY), anf.REPLACE)]
  ```
  only transforms the conditionals of `if` statements (but not, e.g., `while`).

  If no configuration is supplied, the default behavior is to transform all
  expressions except literal constants, which is defined as a configuration as
  ```python
  # For Python 3, and gast library versions before 0.3
  literals = (gast.Num, gast.Str, gast.Bytes, gast.NameConstant)
  [(anf.ASTEdgePattern(anf.ANY, anf.ANY, literals), anf.LEAVE),
   (anf.ASTEdgePattern(anf.ANY, anf.ANY, gast.expr), anf.REPLACE)]
  ```

  Args:
    node: The node to transform.
    ctx: transformer.EntityInfo.  TODO(mdan): What information does this
      argument provide?
    config: Optional ANF configuration.  If omitted, ANF replaces all expression
      expect literal constants.
  """
  return AnfTransformer(ctx, config).visit(node)
