# [MODULE DOC]
# Tests for the activity analysis module. ScopeTest covers the Scope container operations (copy,
# merge, referenced). ActivityAnalyzerTest parses small functions, runs qualified-name resolution
# and activity analysis, then checks the read/modified/bound/param/annotation sets recorded on
# scopes for many statement kinds (imports, calls, loops, ifs, nested functions, lambdas,
# comprehensions, global/nonlocal, annotations, classes). Uses a shared base class with parse and
# assertion helpers. Defines module-level globals used by the global-symbol test.
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
"""Tests for activity module."""

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
from tensorflow.python.platform import test


QN = qual_names.QN
NodeAnno = annos.NodeAnno

global_a = 7
global_b = 17


class ScopeTest(test.TestCase):
  # [CLASS DOC]
  # Unit tests for the Scope container (copy_from, merge_from, copy_of, referenced), plus
  # assertion helpers checking whether a qualified name is read and/or modified in a scope.

  def assertMissing(self, qn, scope):
    # [FUNCTION DOC]
    # HELPER: asserts the name is neither read nor modified in the scope.
    self.assertNotIn(qn, scope.read)
    self.assertNotIn(qn, scope.modified)

  def assertReadOnly(self, qn, scope):
    # [FUNCTION DOC]
    # HELPER: asserts the name is read but not modified in the scope.
    self.assertIn(qn, scope.read)
    self.assertNotIn(qn, scope.modified)

  def assertWriteOnly(self, qn, scope):
    # [FUNCTION DOC]
    # HELPER: asserts the name is modified but not read in the scope.
    self.assertNotIn(qn, scope.read)
    self.assertIn(qn, scope.modified)

  def assertReadWrite(self, qn, scope):
    # [FUNCTION DOC]
    # HELPER: asserts the name is both read and modified in the scope.
    self.assertIn(qn, scope.read)
    self.assertIn(qn, scope.modified)

  def test_copy_from(self):
    # [FUNCTION DOC]
    # Scenario: copy a scope with one modified name into another scope; expects the name to be
    # write-only there. Then add another name to the source and copy back from the copy; expects
    # the extra name to be gone (copy overwrites rather than merges).
    scope = activity.Scope(None)
    scope.modified.add(QN('foo'))
    other = activity.Scope(None)
    other.copy_from(scope)

    self.assertWriteOnly(QN('foo'), other)

    scope.modified.add(QN('bar'))
    scope.copy_from(other)

    self.assertMissing(QN('bar'), scope)

  def test_merge_from(self):
    # [FUNCTION DOC]
    # Scenario: two scopes with overlapping modified/read/bound/deleted names; merge one into the
    # other. Expects the union in all four sets (both names read-write, bound and deleted).
    scope = activity.Scope(None)
    other = activity.Scope(None)

    for col in (scope.modified, scope.read, scope.bound, scope.deleted):
      col.add(QN('foo'))

    for col in (other.modified, other.read, other.bound, other.deleted):
      col.add(QN('foo'))
      col.add(QN('bar'))

    scope.merge_from(other)

    self.assertReadWrite(QN('foo'), scope)
    self.assertReadWrite(QN('bar'), scope)
    self.assertIn(QN('foo'), scope.bound)
    self.assertIn(QN('bar'), scope.bound)
    self.assertIn(QN('foo'), scope.deleted)
    self.assertIn(QN('bar'), scope.deleted)

  def test_copy_of(self):
    # [FUNCTION DOC]
    # Scenario: build copies of a root scope and of a child scope. Expects read names to be
    # preserved in each copy.
    scope = activity.Scope(None)
    scope.read.add(QN('foo'))
    other = activity.Scope.copy_of(scope)

    self.assertReadOnly(QN('foo'), other)

    child_scope = activity.Scope(scope)
    child_scope.read.add(QN('bar'))
    other = activity.Scope.copy_of(child_scope)

    self.assertReadOnly(QN('bar'), other)

  def test_referenced(self):
    # [FUNCTION DOC]
    # Scenario: a chain of three scopes (root, child, non-isolated grandchild) each reading one
    # name, finalized inner to outer. Expects the grandchild and child referenced sets to include
    # names read by all ancestors.
    scope = activity.Scope(None)
    scope.read.add(QN('a'))

    child = activity.Scope(scope)
    child.read.add(QN('b'))

    child2 = activity.Scope(child, isolated=False)
    child2.read.add(QN('c'))

    child2.finalize()
    child.finalize()
    scope.finalize()

    self.assertIn(QN('c'), child2.referenced)
    self.assertIn(QN('b'), child2.referenced)
    self.assertIn(QN('a'), child2.referenced)

    self.assertIn(QN('c'), child.referenced)
    self.assertIn(QN('b'), child.referenced)
    self.assertIn(QN('a'), child.referenced)


class ActivityAnalyzerTestBase(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base for activity analyzer tests; provides parse-and-analyze and scope
  # assertion helpers. Defines no tests itself.

  def _parse_and_analyze(self, test_fn):
    # [FUNCTION DOC]
    # HELPER: parses the source of a function or class into a syntax tree, builds entity info,
    # resolves qualified names and runs activity analysis. Returns the analyzed tree and the
    # entity info.
    # TODO(mdan): Use a custom FunctionTransformer here.
    node, source = parser.parse_entity(test_fn, future_features=())
    entity_info = transformer.EntityInfo(
        name=test_fn.__name__,
        source_code=source,
        source_file=None,
        future_features=(),
        namespace={})
    node = qual_names.resolve(node)
    namer = naming.Namer({})
    ctx = transformer.Context(entity_info, namer, None)
    node = activity.resolve(node, ctx)
    return node, entity_info

  def assertSymbolSetsAre(self, expected, actual, name):
    # [FUNCTION DOC]
    # HELPER: asserts a collection of qualified names, converted to text, equals the expected
    # collection of strings; failure message names the set and lists missing and extra items.
    expected = set(expected)
    actual = set(str(s) for s in actual)
    self.assertSetEqual(
        expected, actual, 'for symbol set: %s\n'
        '  Expected: %s\n'
        '  Got:      %s\n'
        '  Missing:  %s\n'
        '  Extra:    %s\n' % (name.upper(), expected, actual,
                              expected - actual, actual - expected))

  def assertScopeIs(self, scope, used, modified):
    # [FUNCTION DOC]
    # HELPER: asserts the scope's read set equals the expected used names and its modified set
    # equals the expected modified names.
    """Assert the scope contains specific used, modified & created variables."""
    self.assertSymbolSetsAre(used, scope.read, 'read')
    self.assertSymbolSetsAre(modified, scope.modified, 'modified')


class ActivityAnalyzerTest(ActivityAnalyzerTestBase):
  # [CLASS DOC]
  # Tests of the activity analyzer on small parsed functions: checks scope read/modified/bound
  # sets, parameters and annotations recorded for many statement and expression kinds.

  def test_import(self):
    # [FUNCTION DOC]
    # Scenario: import statement with plain, dotted, aliased and dotted-aliased modules. Expects
    # the statement scope to modify the root names a, b and the aliases c, d, reading nothing.

    def test_fn():
      # [FUNCTION DOC]
      # Test subject: performs several imports; never executed.
      import a, b.x, y as c, z.u as d  # pylint:disable=g-multiple-import,g-import-not-at-top,unused-variable

    node, _ = self._parse_and_analyze(test_fn)
    scope = anno.getanno(node.body[0], anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ('a', 'b', 'c', 'd'))

  def test_import_from(self):
    # [FUNCTION DOC]
    # Scenario: two from-imports, one aliased. Expects each statement scope to modify only the
    # imported name (or alias) and read nothing.

    def test_fn():
      # [FUNCTION DOC]
      # Test subject: two from-imports; never executed.
      from x import a  # pylint:disable=g-import-not-at-top,unused-variable
      from y import z as b  # pylint:disable=g-import-not-at-top,unused-variable

    node, _ = self._parse_and_analyze(test_fn)
    scope = anno.getanno(node.body[0], anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ('a',))
    scope = anno.getanno(node.body[1], anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ('b',))

  def test_print_statement(self):
    # [FUNCTION DOC]
    # Scenario: a call to print among assignments. Expects the arguments scope of the print
    # call (or print statement on old Python) to read the two printed variables and modify nothing.

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: assigns two locals, prints one parameter and one local, returns a local.
      b = 0
      c = 1
      print(a, b)
      return c

    node, _ = self._parse_and_analyze(test_fn)
    print_node = node.body[2]
    if isinstance(print_node, gast.Print):
      # Python 2
      print_args_scope = anno.getanno(print_node, NodeAnno.ARGS_SCOPE)
    else:
      # Python 3
      assert isinstance(print_node, gast.Expr)
      # The call node should be the one being annotated.
      print_node = print_node.value
      print_args_scope = anno.getanno(print_node, NodeAnno.ARGS_SCOPE)
    # We basically need to detect which variables are captured by the call
    # arguments.
    self.assertScopeIs(print_args_scope, ('a', 'b'), ())

  def test_call_args(self):
    # [FUNCTION DOC]
    # Scenario: call of an undefined function with two variable arguments. Expects the call's
    # arguments scope to read exactly the two argument variables and modify nothing.

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: calls a function with a parameter and a local, then returns another local.
      b = 0
      c = 1
      foo(a, b)  # pylint:disable=undefined-variable
      return c

    node, _ = self._parse_and_analyze(test_fn)
    call_node = node.body[2].value
    # We basically need to detect which variables are captured by the call
    # arguments.
    self.assertScopeIs(
        anno.getanno(call_node, NodeAnno.ARGS_SCOPE), ('a', 'b'), ())

  def test_call_args_attributes(self):
    # [FUNCTION DOC]
    # Scenario: call passing two attributes of a parameter. Expects the arguments scope to read
    # the parameter and both attribute names, and modify nothing.

    def foo(*_):
      # [FUNCTION DOC]
      # Test helper: accepts any positional arguments and does nothing.
      pass

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: sets an attribute, calls the helper with two attributes, returns another.
      a.c = 0
      foo(a.b, a.c)
      return a.d

    node, _ = self._parse_and_analyze(test_fn)
    call_node = node.body[1].value
    self.assertScopeIs(
        anno.getanno(call_node, NodeAnno.ARGS_SCOPE), ('a', 'a.b', 'a.c'), ())

  def test_call_args_subscripts(self):
    # [FUNCTION DOC]
    # Scenario: call passing a constant-index and a variable-index subscript. Expects the
    # arguments scope to read the container, both subscripts and the index variable only.

    def foo(*_):
      # [FUNCTION DOC]
      # Test helper: accepts any positional arguments and does nothing.
      pass

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: calls the helper with two subscripts of the parameter, returns a third.
      b = 1
      c = 2
      foo(a[0], a[b])
      return a[c]

    node, _ = self._parse_and_analyze(test_fn)
    call_node = node.body[2].value
    self.assertScopeIs(
        anno.getanno(call_node, NodeAnno.ARGS_SCOPE),
        ('a', 'a[0]', 'a[b]', 'b'), ())

  def test_while(self):
    # [FUNCTION DOC]
    # Scenario: while loop that assigns one local and decrements the loop variable. Expects the
    # body scope to read the loop variable and modify it and the local; its parent scope to
    # read and modify the combined names; and the condition scope to read the loop variable only.

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: loops while a local is positive, assigning another local from it.
      b = a
      while b > 0:
        c = b
        b -= 1
      return b, c

    node, _ = self._parse_and_analyze(test_fn)
    while_node = node.body[1]
    self.assertScopeIs(
        anno.getanno(while_node, NodeAnno.BODY_SCOPE), ('b',), ('b', 'c'))
    self.assertScopeIs(
        anno.getanno(while_node, NodeAnno.BODY_SCOPE).parent, ('a', 'b', 'c'),
        ('b', 'c'))
    self.assertScopeIs(
        anno.getanno(while_node, NodeAnno.COND_SCOPE), ('b',), ())

  def test_for(self):
    # [FUNCTION DOC]
    # Scenario: for loop over a parameter. Expects the iterate scope to modify only the loop
    # target, the body scope to read and modify the body variables, and its parent scope to also
    # include the loop target as modified.

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: loops over the parameter, assigning and decrementing locals.
      b = a
      for _ in a:
        c = b
        b -= 1
      return b, c

    node, _ = self._parse_and_analyze(test_fn)
    for_node = node.body[1]
    self.assertScopeIs(
        anno.getanno(for_node, NodeAnno.ITERATE_SCOPE), (), ('_'))
    self.assertScopeIs(
        anno.getanno(for_node, NodeAnno.BODY_SCOPE), ('b',), ('b', 'c'))
    self.assertScopeIs(
        anno.getanno(for_node, NodeAnno.BODY_SCOPE).parent, ('a', 'b', 'c'),
        ('b', 'c', '_'))

  def test_if(self):
    # [FUNCTION DOC]
    # Scenario: if/else assigning different variables in each branch. Expects separate body and
    # else scopes with their own reads/modifications, and parent scopes holding the union of both.

    def test_fn(x):
      # [FUNCTION DOC]
      # Test subject: branches on a parameter, assigning several locals per branch.
      if x > 0:
        x = -x
        y = 2 * x
        z = -y
      else:
        x = 2 * x
        y = -x
        u = -y
      return z, u

    node, _ = self._parse_and_analyze(test_fn)
    if_node = node.body[0]
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE), ('x', 'y'), ('x', 'y', 'z'))
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE).parent, ('x', 'y', 'z', 'u'),
        ('x', 'y', 'z', 'u'))
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE), ('x', 'y'),
        ('x', 'y', 'u'))
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE).parent,
        ('x', 'y', 'z', 'u'), ('x', 'y', 'z', 'u'))

  def test_if_attributes(self):
    # [FUNCTION DOC]
    # Scenario: if/else writing an attribute of a parameter in both branches. Expects attribute
    # writes to be tracked as modified composite names and attribute reads as reads, in both
    # branch scopes and the parent.

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: branches assigning an attribute and a local from attribute reads.
      if a > 0:
        a.b = -a.c
        d = 2 * a
      else:
        a.b = a.c
        d = 1
      return d

    node, _ = self._parse_and_analyze(test_fn)
    if_node = node.body[0]
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE), ('a', 'a.c'), ('a.b', 'd'))
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE), ('a', 'a.c'),
        ('a.b', 'd'))
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE).parent, ('a', 'a.c', 'd'),
        ('a.b', 'd'))

  def test_if_subscripts(self):
    # [FUNCTION DOC]
    # Scenario: if/else writing subscripts of a parameter. Expects subscript writes to be tracked
    # as modified subscript names (not marking the container as modified) in each scope. A TODO
    # notes it is undecided whether such writes should read the container.

    def test_fn(a, b, c, e):
      # [FUNCTION DOC]
      # Test subject: branches assigning subscripts with variable and constant indices.
      if a > 0:
        a[b] = -a[c]
        d = 2 * a
      else:
        a[0] = e
        d = 1
      return d

    node, _ = self._parse_and_analyze(test_fn)
    if_node = node.body[0]
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.BODY_SCOPE), ('a', 'b', 'c', 'a[c]'),
        ('a[b]', 'd'))
    # TODO(mdan): Should subscript writes (a[0] = 1) be considered to read "a"?
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE), ('a', 'e'), ('a[0]', 'd'))
    self.assertScopeIs(
        anno.getanno(if_node, NodeAnno.ORELSE_SCOPE).parent,
        ('a', 'b', 'c', 'd', 'e', 'a[c]'), ('d', 'a[b]', 'a[0]'))

  def test_nested_if(self):
    # [FUNCTION DOC]
    # Scenario: an if/else nested inside an if. Expects the inner statement's body and else
    # scopes each to read the condition variable and modify the assigned variable.

    def test_fn(b):
      # [FUNCTION DOC]
      # Test subject: nested conditionals assigning one local in both inner branches.
      if b > 0:
        if b < 5:
          a = b
        else:
          a = b * b
      return a

    node, _ = self._parse_and_analyze(test_fn)
    inner_if_node = node.body[0].body[0]
    self.assertScopeIs(
        anno.getanno(inner_if_node, NodeAnno.BODY_SCOPE), ('b',), ('a',))
    self.assertScopeIs(
        anno.getanno(inner_if_node, NodeAnno.ORELSE_SCOPE), ('b',), ('a',))

  def test_nested_function(self):
    # [FUNCTION DOC]
    # Scenario: a function defining and calling an inner function. Expects the outer body scope
    # to read the parameter and inner name and modify the inner name; the inner definition's
    # plain scope to modify only its name; and the inner body and args-and-body scopes to read
    # and modify its locals, with its parameter and local bound.

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: defines an inner function squaring its argument and returns its call result.

      def f(x):
        # [FUNCTION DOC]
        # Inner function: squares its argument into a local and returns it.
        y = x * x
        return y

      return f(a)

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'f'), ('f',))

    fn_def_node = node.body[0]

    scope = anno.getanno(fn_def_node, anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ('f'))

    scope = anno.getanno(fn_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('x', 'y'), ('y',))

    scope = anno.getanno(fn_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('x', 'y'), ('y',))
    self.assertSymbolSetsAre(('x', 'y'), scope.bound, 'BOUND')

  def test_nested_lambda(self):
    # [FUNCTION DOC]
    # Scenario: function returning a lambda that uses its own parameter and an outer variable.
    # Expects the outer body and return scopes to read the outer variable only, the lambda body
    # and args-and-body scopes to read both, and the lambda's bound set to hold its parameter.

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: returns a lambda multiplying its argument by the parameter.
      return lambda x: (x * a)

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a',), ())

    return_node = node.body[0]

    scope = anno.getanno(return_node, anno.Static.SCOPE)
    self.assertScopeIs(scope, ('a',), ())

    lam_def_node = return_node.value

    scope = anno.getanno(lam_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'x'), ())

    scope = anno.getanno(lam_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'x'), ())
    self.assertSymbolSetsAre(('x',), scope.bound, 'BOUND')

  def test_nested_function_arg_defaults(self):
    # [FUNCTION DOC]
    # Scenario: inner function whose argument default refers to an outer variable. Expects the
    # definition scope to read that outer variable and modify the function name, while the body
    # scopes only see the inner locals and parameter.

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: defines an inner function with a default argument and calls it.

      def f(x=a):
        # [FUNCTION DOC]
        # Inner function: squares its (defaulted) argument into a local and returns it.
        y = x * x
        return y

      return f(a)

    node, _ = self._parse_and_analyze(test_fn)
    fn_def_node = node.body[0]

    self.assertScopeIs(
        anno.getanno(fn_def_node, anno.Static.SCOPE), ('a',), ('f',))

    scope = anno.getanno(fn_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('x', 'y'), ('y',))

    scope = anno.getanno(fn_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('x', 'y'), ('y',))
    self.assertSymbolSetsAre(('x', 'y'), scope.bound, 'BOUND')

  def test_constructor_attributes(self):
    # [FUNCTION DOC]
    # Scenario: analyze a class whose constructor sets an attribute of self and a nested
    # attribute. Expects the constructor body scope to read self, the argument and self's
    # attribute, and to mark self, its attribute and the nested attribute as modified.

    class TestClass(object):
      # [CLASS DOC]
      # Test subject: class with a constructor assigning attributes on self.

      def __init__(self, a):
        # [FUNCTION DOC]
        # Test subject: stores the argument as an attribute and sets a nested attribute.
        self.b = a
        self.b.c = 1

    node, _ = self._parse_and_analyze(TestClass)
    init_node = node.body[0]
    self.assertScopeIs(
        anno.getanno(init_node, NodeAnno.BODY_SCOPE), ('self', 'a', 'self.b'),
        ('self', 'self.b', 'self.b.c'))

  def test_aug_assign_subscripts(self):
    # [FUNCTION DOC]
    # Scenario: augmented assignment to a subscript. Expects the body scope to read the
    # container and the subscript and modify only the subscript.

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: increments an element of the parameter.
      a[0] += 1

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('a', 'a[0]'), ('a[0]',))

  def test_return_vars_are_read(self):
    # [FUNCTION DOC]
    # Scenario: function returning one parameter. Expects both the body scope and the return
    # statement scope to read only that parameter and modify nothing.

    def test_fn(a, b, c):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: returns its third parameter.
      return c

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('c',), ())
    self.assertScopeIs(
        anno.getanno(node.body[0], anno.Static.SCOPE), ('c',), ())

  def test_raise_names_are_read(self):
    # [FUNCTION DOC]
    # Scenario: function raising one parameter. Expects the body scope and the raise statement
    # scope to read only that parameter and modify nothing.

    def test_fn(a, b, c):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: raises its second parameter.
      raise b

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('b',), ())
    self.assertScopeIs(
        anno.getanno(node.body[0], anno.Static.SCOPE), ('b',), ())

  def test_except_exposes_names(self):
    # [FUNCTION DOC]
    # Scenario: bare except handler assigning a variable. Expects the assignment to be visible
    # in the function body scope (reads c, modifies b), i.e. handler names leak outward.

    def test_fn(a, b, c):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: try block with a bare handler assigning one parameter from another.
      try:
        pass
      except:  # pylint: disable=bare-except
        b = c

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('c',), ('b',))

  def test_except_hides_exception_var_name(self):
    # [FUNCTION DOC]
    # Scenario: handler binding the exception to a name and using it. Expects the body scope to
    # read the exception type expression and modify the assigned variable, with the exception
    # variable name hidden (neither read nor modified outside the handler).

    def test_fn(a, b, c):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: handler catching a parameter as an exception variable and assigning it.
      try:
        pass
      except a as e:
        b = e

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('a',), ('b',))

  def test_aug_assign(self):
    # [FUNCTION DOC]
    # Scenario: augmented assignment of one parameter to another. Expects the target to be both
    # read and modified and the value to be read.

    def test_fn(a, b):
      # [FUNCTION DOC]
      # Test subject: adds the second parameter into the first.
      a += b

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('a', 'b'), ('a'))

  def test_aug_assign_rvalues(self):
    # [FUNCTION DOC]
    # Scenario: augmented assignment whose target is a subscript of a call result. Expects only
    # the called function name and the added value to be read, and nothing modified (the target
    # has no qualified name).

    a = dict(bar=3)

    def foo():
      # [FUNCTION DOC]
      # Test helper: returns the enclosing test's dictionary.
      return a

    def test_fn(x):
      # [FUNCTION DOC]
      # Test subject: adds its argument into an element of the helper's returned mapping.
      foo()['bar'] += x

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    self.assertScopeIs(
        anno.getanno(fn_node, NodeAnno.BODY_SCOPE), ('foo', 'x'), ())

  def test_lambda(self):
    # [FUNCTION DOC]
    # Scenario: function returning a lambda with no parameters that uses two outer variables.
    # Expects the free variables to be read in the body scope, the lambda's own plain scope to
    # be empty, its body and args-and-body scopes to read both, nothing bound, no parameters.

    def test_fn(a, b):
      # [FUNCTION DOC]
      # Test subject: returns a parameterless lambda adding the two parameters.
      return lambda: (a + b)

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())

    lam_def_node = node.body[0].value

    scope = anno.getanno(lam_def_node, anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ())

    scope = anno.getanno(lam_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())

    scope = anno.getanno(lam_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())
    self.assertSymbolSetsAre((), scope.bound, 'BOUND')

    scope = anno.getanno(lam_def_node.args, anno.Static.SCOPE)
    self.assertSymbolSetsAre((), scope.params.keys(), 'lambda params')

  def test_lambda_params_args(self):
    # [FUNCTION DOC]
    # Scenario: lambda whose parameter shadows an outer name. Expects the outer body scope to read
    # only the unshadowed outer variable, the lambda scopes to read both names, bound set to
    # contain the lambda parameter, and the parameters mapping to list it.

    def test_fn(a, b):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: returns a lambda taking a parameter and adding an outer variable.
      return lambda a: a + b

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    # Note: `a` in `a + b` is not "read" here because it's hidden by the `a`
    # argument.
    self.assertScopeIs(scope, ('b',), ())

    lam_def_node = node.body[0].value

    scope = anno.getanno(lam_def_node, anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ())

    scope = anno.getanno(lam_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())

    scope = anno.getanno(lam_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())
    self.assertSymbolSetsAre(('a',), scope.bound, 'BOUND')

    scope = anno.getanno(lam_def_node.args, anno.Static.SCOPE)
    self.assertSymbolSetsAre(('a',), scope.params.keys(), 'lambda params')

  def test_lambda_params_arg_defaults(self):
    # [FUNCTION DOC]
    # Scenario: lambda with a parameter defaulting to an outer variable. Expects the default to
    # be read in the lambda's definition scope and the outer body, the shadowed name not to be
    # read outside, and the parameter to be bound inside the lambda.

    def test_fn(a, b, c):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: returns a lambda with a defaulted parameter adding an outer variable.
      return lambda b=c: a + b

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    # Note: `b` is not "read" here because it's hidden by the argument.
    self.assertScopeIs(scope, ('a', 'c'), ())

    lam_def_node = node.body[0].value

    scope = anno.getanno(lam_def_node, anno.Static.SCOPE)
    self.assertScopeIs(scope, ('c',), ())

    scope = anno.getanno(lam_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())

    scope = anno.getanno(lam_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b'), ())
    self.assertSymbolSetsAre(('b',), scope.bound, 'BOUND')

    scope = anno.getanno(lam_def_node.args, anno.Static.SCOPE)
    self.assertSymbolSetsAre(('b',), scope.params.keys(), 'lambda params')

  def test_lambda_complex(self):
    # [FUNCTION DOC]
    # Scenario: immediately-called lambda with three parameters (one defaulted) whose result is
    # added to an outer variable and assigned. Expects correct outer reads and modification, and
    # the lambda's scopes, bound names and parameter names to match its three parameters.

    def test_fn(a, b, c, d, e):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: assigns the result of calling a lambda plus a parameter.
      a = (lambda a, b, c=e: a + b + c)(d, 1, 2) + b

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('d', 'b', 'e'), ('a',))

    lam_def_node = node.body[0].value.left.func

    scope = anno.getanno(lam_def_node, anno.Static.SCOPE)
    self.assertScopeIs(scope, ('e',), ())

    scope = anno.getanno(lam_def_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b', 'c'), ())

    scope = anno.getanno(lam_def_node, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b', 'c'), ())
    self.assertSymbolSetsAre(('a', 'b', 'c'), scope.bound, 'BOUND')

    scope = anno.getanno(lam_def_node.args, anno.Static.SCOPE)
    self.assertSymbolSetsAre(
        ('a', 'b', 'c'), scope.params.keys(), 'lambda params')

  def test_lambda_nested(self):
    # [FUNCTION DOC]
    # Scenario: a lambda containing another lambda inside a call argument, with a default value
    # from an outer variable. Expects each level's definition, body and args-and-body scopes,
    # bound names and parameter names to reflect shadowing and propagation of free reads outward.

    def test_fn(a, b, c, d, e, f):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: assigns a lambda whose body calls a function with an inner lambda.
      a = lambda a, b: d(lambda b=f: a + b + c)  # pylint: disable=undefined-variable

    node, _ = self._parse_and_analyze(test_fn)

    fn_node = node
    scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('d', 'c', 'f'), ('a',))

    outer_lam_def = node.body[0].value

    scope = anno.getanno(outer_lam_def, anno.Static.SCOPE)
    self.assertScopeIs(scope, (), ())

    scope = anno.getanno(outer_lam_def, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('d', 'f', 'a', 'c'), ())

    scope = anno.getanno(outer_lam_def, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('d', 'f', 'a', 'c'), ())
    self.assertSymbolSetsAre(('a', 'b'), scope.bound, 'BOUND')

    scope = anno.getanno(outer_lam_def.args, anno.Static.SCOPE)
    self.assertSymbolSetsAre(('a', 'b'), scope.params.keys(), 'lambda params')

    inner_lam_def = outer_lam_def.body.args[0]

    scope = anno.getanno(inner_lam_def, anno.Static.SCOPE)
    self.assertScopeIs(scope, ('f',), ())

    scope = anno.getanno(inner_lam_def, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b', 'c'), ())

    scope = anno.getanno(inner_lam_def, NodeAnno.ARGS_AND_BODY_SCOPE)
    self.assertScopeIs(scope, ('a', 'b', 'c'), ())
    self.assertSymbolSetsAre(('b',), scope.bound, 'BOUND')

    scope = anno.getanno(inner_lam_def.args, anno.Static.SCOPE)
    self.assertSymbolSetsAre(('b',), scope.params.keys(), 'lambda params')

  def test_comprehension_targets_are_isolated(self):
    # [FUNCTION DOC]
    # Scenario: set comprehension over a parameter. Expects the comprehension target not to be
    # read or modified in the function scope (only the iterable read and result variable modified).

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: assigns a set comprehension over the parameter.
      b = {c for c in a}  # pylint:disable=unused-variable

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('a',), ('b',))

  def test_comprehension_targets_are_isolated_list_function_w_generator(self):
    # [FUNCTION DOC]
    # Scenario: a generator expression passed to the list function. Expects the target hidden,
    # and the iterable plus the list function name read.

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: assigns the list built from a generator expression over the parameter.
      b = list(c for c in a)  # pylint:disable=unused-variable

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('a', 'list'), ('b',))

  def test_list_comprehension_targets_are_sometimes_isolated(self):
    # [FUNCTION DOC]
    # Scenario: list comprehension over a parameter. Expects the target hidden (only the iterable
    # read and result variable modified).

    def test_fn(a):
      # [FUNCTION DOC]
      # Test subject: assigns a list comprehension over the parameter.
      b = [c for c in a]  # pylint:disable=unused-variable

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('a',), ('b',))

  def test_comprehension_targets_are_isolated_in_augassign(self):
    # [FUNCTION DOC]
    # Scenario: augmented assignment of a list comprehension. Expects the comprehension target
    # hidden, the iterable and augmented target read, and the augmented target modified.

    def test_fn(a, b):
      # [FUNCTION DOC]
      # Test subject: adds a list comprehension over one parameter into the other.
      b += [c for c in a]  # pylint:disable=unused-variable

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('a', 'b'), ('b',))

  def test_comprehension_generator_order(self):
    # [FUNCTION DOC]
    # Scenario: dictionary comprehension with two generators where the second iterates over a
    # variable bound by the first. Expects only the outermost iterable to be read and only the
    # result variable modified, showing generators are processed before the element expressions.

    def test_fn(a, b, c):  # pylint:disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: assigns a dictionary comprehension with chained generators.
      e = {d: (a, b) for (a, b) in c for d in b}  # pylint:disable=unused-variable,g-complex-comprehension

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('c',), ('e',))

  def test_global_symbol(self):
    # [FUNCTION DOC]
    # Scenario: function declaring two module globals and assigning one from the other plus a
    # parameter. Expects the body scope to read both globals and the parameter, modify the
    # assigned global, list both globals in the globals set, and the declaration statement scope
    # to read its declared name.

    def test_fn(c):
      # [FUNCTION DOC]
      # Test subject: declares two globals and assigns one from the other and a parameter.
      global global_a
      global global_b
      global_a = global_b + c

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('global_a', 'global_b', 'c'), ('global_a',))
    self.assertSetEqual(body_scope.globals, set(
        (QN('global_a'), QN('global_b'))))
    global_a_scope = anno.getanno(fn_node.body[0], anno.Static.SCOPE)
    self.assertScopeIs(global_a_scope, ('global_a',), ())

  def test_nonlocal_symbol(self):
    # [FUNCTION DOC]
    # Scenario: function declaring two enclosing-scope variables nonlocal and assigning one from
    # the other plus a parameter. Expects the body scope to read all three names and modify the
    # assigned one; the nonlocal statement scope reads its declared name.
    nonlocal_a = 3
    nonlocal_b = 13

    def test_fn(c):
      # [FUNCTION DOC]
      # Test subject: declares two nonlocals and assigns one from the other and a parameter.
      nonlocal nonlocal_a
      nonlocal nonlocal_b
      nonlocal_a = nonlocal_b + c

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(
        body_scope, ('nonlocal_a', 'nonlocal_b', 'c'), ('nonlocal_a',))
    nonlocal_a_scope = anno.getanno(fn_node.body[0], anno.Static.SCOPE)
    self.assertScopeIs(nonlocal_a_scope, ('nonlocal_a',), ())

  def test_annotated_assign(self):
    # [FUNCTION DOC]
    # Scenario: annotated assignment with a variable used as the annotation. Expects the
    # annotation symbol to be read and recorded in the annotations set of both the body scope and
    # the statement scope, with the target modified and the value read.
    b = int

    def test_fn(c):
      # [FUNCTION DOC]
      # Test subject: annotated assignment of a parameter, then returns the target.
      a: b = c
      return a

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node

    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('b', 'c', 'a'), ('a',))
    self.assertSymbolSetsAre(('b',), body_scope.annotations, 'annotations')

    ann_assign_scope = anno.getanno(fn_node.body[0], anno.Static.SCOPE)
    self.assertScopeIs(ann_assign_scope, ('b', 'c'), ('a',))
    self.assertSymbolSetsAre(
        ('b',), ann_assign_scope.annotations, 'annotations')

  def test_pure_definition(self):
    # [FUNCTION DOC]
    # Scenario: a bare annotated declaration without a value. Expects the annotation symbol to be
    # read and recorded as annotation, and the target to be reported as modified.
    b = int

    def test_fn():
      # [FUNCTION DOC]
      # Test subject: declares an annotated name without a value and returns it.
      a: b
      return a

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node

    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('b', 'a'), ('a',))
    self.assertSymbolSetsAre(('b',), body_scope.annotations, 'annotations')

    ann_assign_scope = anno.getanno(fn_node.body[0], anno.Static.SCOPE)
    self.assertScopeIs(ann_assign_scope, ('b',), ('a',))
    self.assertSymbolSetsAre(
        ('b',), ann_assign_scope.annotations, 'annotations')

  def test_function_def_annotations(self):
    # [FUNCTION DOC]
    # Scenario: function with parameter and return annotations from outer variables. Expects the
    # definition scope to read both annotation symbols, record them as annotations and modify the
    # function name, while the body scope reads only the parameter with no annotations.
    b = int
    c = int

    def test_fn(a: b) -> c:
      # [FUNCTION DOC]
      # Test subject: annotated function returning its parameter.
      return a

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node

    fn_scope = anno.getanno(fn_node, anno.Static.SCOPE)
    self.assertScopeIs(fn_scope, ('b', 'c'), ('test_fn',))
    self.assertSymbolSetsAre(('b', 'c'), fn_scope.annotations, 'annotations')

    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('a',), ())
    self.assertSymbolSetsAre((), body_scope.annotations, 'annotations')

  def test_class_definition_basic(self):
    # [FUNCTION DOC]
    # Scenario: a class defined inside a function with a base computed from a call. Expects the
    # function body scope to read the call pieces and class name and modify the class name.

    def test_fn(a, b):
      # [FUNCTION DOC]
      # Test subject: defines a class deriving from a call result and returns it.
      class C(a(b)):
        # [CLASS DOC]
        # Test subject: class deriving from a call result with one class attribute.
        d = 1
      return C

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('a', 'b', 'C'), ('C',))

  def test_class_definition_isolates_method_writes(self):
    # [FUNCTION DOC]
    # Scenario: class with an attribute and a method that writes a local and reads an outer
    # variable. Expects the outer body scope to read the base pieces, class name and the outer
    # variable used in the method, and to modify only the class name (class and method locals
    # do not leak).

    def test_fn(a, b, c):
      # [FUNCTION DOC]
      # Test subject: defines a class with a method and returns the class.
      class C(a(b)):
        # [CLASS DOC]
        # Test subject: class deriving from a call result with a class attribute and a method.
        d = 1

        def e(self):
          # [FUNCTION DOC]
          # Test subject: computes a local from an outer variable and returns it.
          f = c + 1
          return f
      return C

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('a', 'b', 'C', 'c'), ('C',))


if __name__ == '__main__':
  test.main()
