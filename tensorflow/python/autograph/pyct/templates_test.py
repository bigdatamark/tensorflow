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
# Unit tests for the templates module: placeholder substitution (names, tuples, function names,
# attributes, keywords, code blocks, expressions) and correct read/write context assignment on
# substituted nodes. Also defines small private helpers (a context clearer, a context checker and
# a parse function) used by the tests. Run as a test binary; no import-time side effects.
"""Tests for templates module."""

import types

from absl.testing import parameterized
import gast

from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names as qn
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.platform import test


class _CtxClearer(gast.NodeTransformer):
  # [CLASS DOC]
  # HELPER: test utility that wipes the context field of every node in a tree, simulating
  # nodes built without a known read/write context.

  def visit(self, node):
    # [FUNCTION DOC]
    # HELPER: visits children first, then sets this node's context (if it has one) to nothing,
    # mutating the tree in place. Returns the node itself.
    super(_CtxClearer, self).visit(node)
    if hasattr(node, 'ctx'):
      node.ctx = None
    return node


def _parse_with_unset_ctx(expr_source):
  # [FUNCTION DOC]
  # HELPER: builds test inputs. Parses an expression string into an AST node and clears the
  # context of all nodes in it. Returns the node.
  ast_node = parser.parse_expression(expr_source)
  _CtxClearer().visit(ast_node)
  return ast_node


class _CtxChecker(gast.NodeTransformer):
  # [CLASS DOC]
  # HELPER: test utility that walks a tree and asserts, through a test case instance, that the
  # top node has an expected context class and every other node with a context is a load.

  def __init__(self, test_instance, expected_ctx):
    # [FUNCTION DOC]
    # Stores the test case used for assertions, the expected context class for the top node, and
    # a flag marking that the top node has not been seen yet.
    self.at_top_level = True
    self.test_instance = test_instance
    self.expected_ctx = expected_ctx

  def visit(self, node):
    # [FUNCTION DOC]
    # Asserts that the node's context (if any) is an instance of the currently expected class.
    # After the first call, the expectation becomes load for all descendants. Then continues the
    # normal traversal and returns its result.
    if hasattr(node, 'ctx'):
      self.test_instance.assertIsInstance(node.ctx, self.expected_ctx)
    if self.at_top_level:
      self.at_top_level = False
      self.expected_ctx = gast.Load
    return super(_CtxChecker, self).visit(node)


class TemplatesTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for templates.replace and replace_as_expression. Mixes the TensorFlow test case with
  # parameterized test support. Most tests define a template string, substitute placeholders,
  # then either load the generated function and call it, or inspect context fields of the AST.

  def assertExpectedCtxSet(self, node, ctx):
    # [FUNCTION DOC]
    # HELPER: custom assertion for the context tests. Walks the node with a context checker
    # asserting the top node has context class ctx and everything below it is a load.
    """Assert that node has ctx=ctx at top and ctx=gast.Load everywhere else."""
    checker = _CtxChecker(self, ctx)
    checker.visit(node)

  def test_replace_tuple(self):
    # [FUNCTION DOC]
    # Template returns a one-element tuple of a placeholder; replaces it with a tuple of the two
    # parameter names, loads the function and asserts calling it with 2 and 3 returns (2, 3).
    template = """
      def test_fn(a, c):
        return b,
    """

    node = templates.replace(template, b=('a', 'c'))[0]
    result, _, _ = loader.load_ast(node)

    self.assertEqual((2, 3), result.test_fn(2, 3))

  def test_replace_variable(self):
    # [FUNCTION DOC]
    # Replaces the parameter-name placeholder with another name everywhere it appears (both
    # assignments and the return), loads the function and asserts calling with 2 gives 7.
    template = """
      def test_fn(a):
        a += 1
        a = 2 * a + 1
        return b
    """

    node = templates.replace(template, a='b')[0]
    result, _, _ = loader.load_ast(node)
    self.assertEqual(7, result.test_fn(2))

  def test_replace_function_name(self):
    # [FUNCTION DOC]
    # Replaces the placeholder function name with a real name, loads it and asserts calling it
    # with 2 returns 7.
    template = """
      def fname(a):
        a += 1
        a = 2 * a + 1
        return a
    """

    node = templates.replace(template, fname='test_fn')[0]
    result, _, _ = loader.load_ast(node)
    self.assertEqual(7, result.test_fn(2))

  def test_replace_code_block(self):
    # [FUNCTION DOC]
    # Replaces a statement placeholder with a list of two identical assignment statements whose
    # name nodes have a bogus context class. Asserts the generated function loads and, called
    # with 1, returns 3, showing the contexts were fixed up and both statements were inserted.
    template = """
      def test_fn(a):
        block
        return a
    """

    class ShouldBeReplaced(object):
      # [CLASS DOC]
      # Dummy marker class used as an invalid context value in the replacement nodes; the
      # template machinery is expected to overwrite it.
      pass

    node = templates.replace(
        template,
        block=[
            gast.Assign(
                targets=[
                    gast.Name(
                        'a',
                        ctx=ShouldBeReplaced,
                        annotation=None,
                        type_comment=None)
                ],
                value=gast.BinOp(
                    gast.Name(
                        'a',
                        ctx=ShouldBeReplaced,
                        annotation=None,
                        type_comment=None), gast.Add(),
                    gast.Constant(1, kind=None)),
            ),
        ] * 2)[0]
    result, _, _ = loader.load_ast(node)
    self.assertEqual(3, result.test_fn(1))

  def test_replace_attribute(self):
    # [FUNCTION DOC]
    # Replaces an attribute-name placeholder with a name; loads the function and asserts it reads
    # that attribute from a module object (value 3). Also asserts that replacing the attribute
    # with a number raises a value error.
    template = """
      def test_fn(a):
        return a.foo
    """

    node = templates.replace(template, foo='b')[0]
    result, _, _ = loader.load_ast(node)
    mod = types.ModuleType('test')
    mod.b = 3
    self.assertEqual(3, result.test_fn(mod))

    with self.assertRaises(ValueError):
      templates.replace(template, foo=1)

  def test_replace_attribute_context(self):
    # [FUNCTION DOC]
    # Replaces an assignment target with a dotted-name expression and asserts the outer attribute
    # gets a store context while the inner attribute and base name get load contexts.
    template = """
      def test_fn(foo):
        foo = 0
    """

    node = templates.replace(
        template,
        foo=parser.parse_expression('a.b.c'))[0]
    self.assertIsInstance(node.body[0].targets[0].ctx, gast.Store)
    self.assertIsInstance(node.body[0].targets[0].value.ctx, gast.Load)
    self.assertIsInstance(node.body[0].targets[0].value.value.ctx, gast.Load)

  def test_replace_list_context(self):
    # [FUNCTION DOC]
    # Replaces an assignment target with a list display of two names and asserts the list and
    # both elements get a store context.
    template = """
      def test_fn(foo):
        foo = 0
    """

    node = templates.replace(template, foo=parser.parse_expression('[a, b]'))[0]
    self.assertIsInstance(node.body[0].targets[0].ctx, gast.Store)
    self.assertIsInstance(node.body[0].targets[0].elts[0].ctx, gast.Store)
    self.assertIsInstance(node.body[0].targets[0].elts[1].ctx, gast.Store)

  def test_replace_tuple_context(self):
    # [FUNCTION DOC]
    # Same as the list case but with a tuple of two names: asserts store context on the tuple and
    # both elements.
    template = """
      def test_fn(foo):
        foo = 0
    """

    node = templates.replace(template, foo=parser.parse_expression('(a, b)'))[0]
    self.assertIsInstance(node.body[0].targets[0].ctx, gast.Store)
    self.assertIsInstance(node.body[0].targets[0].elts[0].ctx, gast.Store)
    self.assertIsInstance(node.body[0].targets[0].elts[1].ctx, gast.Store)

  def test_replace_expression_context(self):
    # [FUNCTION DOC]
    # Replaces a bare expression statement placeholder with an arithmetic expression and asserts
    # nested operand names have load context.
    template = """
      def test_fn():
        foo
    """

    node = templates.replace(
        template, foo=parser.parse_expression('a + 2 * b / -c'))[0]
    self.assertIsInstance(node.body[0].left.ctx, gast.Load)
    self.assertIsInstance(node.body[0].right.left.right.ctx, gast.Load)

  def test_replace_complex_context(self):
    # [FUNCTION DOC]
    # Replaces an assignment target with an attribute of a call whose argument is a tuple holding
    # a list. Asserts the outermost target is stored while the call argument tuple, the nested
    # list and its elements are loads.
    template = """
      def test_fn():
        foo = 0
    """

    node = templates.replace(
        template, foo=parser.parse_expression('bar(([a, b],)).baz'))[0]
    self.assertIsInstance(node.body[0].targets[0].ctx, gast.Store)
    function_call_arg = node.body[0].targets[0].value.args[0]
    self.assertIsInstance(function_call_arg.elts[0].ctx, gast.Load)
    self.assertIsInstance(function_call_arg.elts[0].elts[0].ctx, gast.Load)
    self.assertIsInstance(function_call_arg.elts[0].elts[1].ctx, gast.Load)

  def test_replace_index(self):
    # [FUNCTION DOC]
    # Replaces an assignment target with an attribute of a call whose argument is a subscript.
    # Asserts the subscript and its index have load contexts.
    template = """
      def test_fn():
        foo = 0
    """

    node = templates.replace(
        template, foo=parser.parse_expression('foo(a[b]).bar'))[0]
    function_call_arg = node.body[0].targets[0].value.args[0]
    self.assertIsInstance(function_call_arg.ctx, gast.Load)
    self.assertIsInstance(function_call_arg.slice.ctx, gast.Load)

  def test_replace_call_keyword(self):
    # [FUNCTION DOC]
    # Replaces a keyword placeholder in a call with two parsed keyword arguments, loads the
    # function and asserts the result is 9. Then asserts replacing with an empty list raises a
    # value error.
    # KNOWN ISSUE: the last statement (replacing with a number) is never reached because the
    # preceding call already raises inside the same assertion block, so that case is untested.
    template = """
      def test_fn():
        def f(a, d, f):
          return a + d + f
        return f(1, kws=None)
    """

    source = parser.parse_expression('f(d=3, f=5)')
    node = templates.replace(template, kws=source.keywords)[0]
    result, _, _ = loader.load_ast(node)
    self.assertEqual(9, result.test_fn())

    with self.assertRaises(ValueError):
      templates.replace(template, kws=[])
      templates.replace(template, kws=1)

  def test_replace_name_with_call(self):
    # [FUNCTION DOC]
    # Replaces a returned name with a parsed call-of-call expression that uses local helpers;
    # loads the function and asserts the result is 15.
    template = """
      def test_fn():
        b = 5
        def g(a):
          return 3 * a
        def f():
          return g
        return foo
    """

    source = parser.parse_expression('f()(b)')
    node = templates.replace(template, foo=source)[0]
    result, _, _ = loader.load_ast(node)
    self.assertEqual(15, result.test_fn())

  def test_replace_name_with_dict(self):
    # [FUNCTION DOC]
    # Replaces the subscripted name with a parsed dictionary display, loads the function and
    # asserts looking up the key yields 3.
    template = """
      def test_fn():
        return foo['bar']
    """

    source = parser.parse_expression('{\'bar\': 3}')
    node = templates.replace(template, foo=source)[0]
    result, _, _ = loader.load_ast(node)
    self.assertEqual(3, result.test_fn())

  def test_replace_as_expression(self):
    # [FUNCTION DOC]
    # Uses the expression variant on a call template with string replacements and asserts the
    # result is a call node with the replaced callee name and first argument name.
    template = """
      foo(a)
    """

    node = templates.replace_as_expression(template, foo='bar', a='baz')
    self.assertIsInstance(node, gast.Call)
    self.assertEqual(node.func.id, 'bar')
    self.assertEqual(node.args[0].id, 'baz')

  def test_replace_as_expression_restrictions(self):
    # [FUNCTION DOC]
    # Asserts that a two-statement template passed to the expression variant raises a value
    # error.
    template = """
      foo(a)
      bar(b)
    """
    with self.assertRaises(ValueError):
      templates.replace_as_expression(template)

  def test_function_call_in_list(self):
    # [FUNCTION DOC]
    # Regression-style smoke test: substitutes a list display containing nested calls into a
    # call template using the expression variant; passes if no exception is raised.
    template = """
        foo(bar)
    """
    source = parser.parse_expression('[a(b(1))]')
    templates.replace_as_expression(template, bar=source)

  def test_star_comprehension_in_function_call(self):
    # [FUNCTION DOC]
    # Substitutes a callee and an argument list taken from a parsed call whose argument is a
    # starred list comprehension. Asserts the comprehension target has store context and its
    # element expression has load context.
    template = """
      a = foo(func, args)
    """
    source = parser.parse_expression('bar(*[i for i in range(j)])')
    node = templates.replace(template, func=source.func, args=source.args)
    arg_node = node[0].value.args[1].value
    self.assertIsInstance(arg_node.generators[0].target.ctx, gast.Store)
    self.assertIsInstance(arg_node.elt.ctx, gast.Load)

  def test_lambda_in_function_call(self):
    # [FUNCTION DOC]
    # Substitutes a list holding a lambda into a call template. Asserts the lambda parameter
    # keeps a parameter context and the lambda body name has load context.
    template = """
      a = foo(arg)
    """
    source = parser.parse_expression('[lambda i: i]')
    node = templates.replace(template, arg=source)
    lambda_arg = node[0].value.args[0].elts[0]
    self.assertIsInstance(lambda_arg.args.args[0].ctx, gast.Param)
    self.assertIsInstance(lambda_arg.body.ctx, gast.Load)

  def test_replace_name_with_subscript(self):
    # [FUNCTION DOC]
    # Replaces an assignment target with a qualified name denoting a dictionary subscript and
    # asserts the resulting target has store context and its base has load context.
    template = """
        foo = bar
    """
    replacement = qn.QN(qn.QN('dictionary'), subscript=qn.QN('key'))

    node = templates.replace(template, foo=replacement)[0].targets[0]
    self.assertIsInstance(node.ctx, gast.Store)
    self.assertIsInstance(node.value.ctx, gast.Load)

  @parameterized.named_parameters([
      ('mixed_attr_subscript', 'a.b["c"]'),
      ('mixed_subscript_attr', 'a[b.c]'),
      ('nested_subscript', 'a[b[c]]'),
      ('repeated_subscript', 'a[b][c]'),
  ])
  def test_replace_name_mixed_attr_subscript(self, expression_source):
    # [FUNCTION DOC]
    # Parameterized over four expression strings mixing attributes and subscripts. Parses the
    # expression with contexts cleared, substitutes it once as an assignment target (asserting
    # store at the top and load below) and once as the assigned value (asserting load throughout).
    template = 'foo = bar'
    replacement = _parse_with_unset_ctx(expression_source)

    target_node = templates.replace(template, foo=replacement)[0].targets[0]
    self.assertExpectedCtxSet(target_node, gast.Store)

    value_node = templates.replace(template, bar=replacement)[0].value
    self.assertExpectedCtxSet(value_node, gast.Load)

if __name__ == '__main__':
  test.main()
