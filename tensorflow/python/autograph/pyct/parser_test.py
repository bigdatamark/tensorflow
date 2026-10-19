# [MODULE DOC]
# Unit tests for the autograph parser module. Covers parsing live functions and lambdas
# (including lambda disambiguation by location and by signature, ambiguous lambdas, multi-line
# lambdas, lambdas inside larger expressions and call arguments), parsing functions with
# comments and multi-line strings, block dedenting (basic, out-of-line comments, multi-line
# strings and expressions, continuations), parse_expression, unparse and a round trip of
# extended slices. Defines a single test case class.
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
"""Tests for parser module."""

import re
import sys
import textwrap

import gast

from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.platform import test


class ParserTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the parser module's parse, dedent, expression-parse and unparse functions.
  # Provides two custom assertion helpers and an evaluation helper; tests share no state.

  def assertAstMatches(self, actual_node, expected_node_src, expr=True):
    # [FUNCTION DOC]
    # HELPER: custom assertion used by the parser tests.
    # Parses expected_node_src (text) and asserts the actual tree structurally matches it. If expr
    # is true (default) the text is wrapped in parentheses (so multi-line expressions parse) and
    # the expression value is compared; otherwise the first statement is compared. On failure the
    # message shows the pretty-printed expected and actual trees. Returns nothing.
    if expr:
      # Ensure multi-line expressions parse.
      expected_node = gast.parse('({})'.format(expected_node_src)).body[0]
      expected_node = expected_node.value
    else:
      expected_node = gast.parse(expected_node_src).body[0]

    msg = 'AST did not match expected:\n{}\nActual:\n{}'.format(
        pretty_printer.fmt(expected_node),
        pretty_printer.fmt(actual_node))
    self.assertTrue(ast_util.matches(actual_node, expected_node), msg)

  def test_parse_entity(self):
    # [FUNCTION DOC]
    # Parses a simple local function with no future features and asserts the resulting node's
    # name is the function's name.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function to be parsed; returns its argument plus one.
      return x + 1

    node, _ = parser.parse_entity(f, future_features=())
    self.assertEqual('f', node.name)

  def test_parse_lambda(self):
    # [FUNCTION DOC]
    # Parses a simple one-line lambda and asserts the node matches both the returned source text
    # and the expected lambda expression.

    l = lambda x: x + 1
    expected_node_src = 'lambda x: (x + 1)'

    node, source = parser.parse_entity(l, future_features=())
    self.assertAstMatches(node, source)
    self.assertAstMatches(node, expected_node_src)

  def test_parse_lambda_prefix_cleanup(self):
    # [FUNCTION DOC]
    # Same as the simple lambda test but the lambda's variable name begins with the lambda
    # keyword; asserts the prefix is not confused with the keyword and the node still matches.

    lambda_lam = lambda x: x + 1
    expected_node_src = 'lambda x: (x + 1)'

    node, source = parser.parse_entity(lambda_lam, future_features=())
    self.assertAstMatches(node, source)
    self.assertAstMatches(node, expected_node_src)

  def test_parse_lambda_resolution_by_location(self):
    # [FUNCTION DOC]
    # Defines three identical lambdas on consecutive lines and parses the middle one; asserts the
    # node matches and the returned source is exactly that one lambda's text (the right one is
    # chosen by line location).

    _ = lambda x: x + 1
    l = lambda x: x + 1
    _ = lambda x: x + 1
    expected_node_src = 'lambda x: (x + 1)'

    node, source = parser.parse_entity(l, future_features=())
    self.assertAstMatches(node, source)
    self.assertAstMatches(node, expected_node_src)
    self.assertEqual(source, 'lambda x: x + 1')

  def test_parse_lambda_resolution_by_signature(self):
    # [FUNCTION DOC]
    # Defines a lambda returning a two-parameter lambda on one line. Parses the outer one and then
    # the inner one (obtained by calling the outer); asserts each parse picks the lambda whose
    # parameter list matches, with the expected tree and source text.

    l = lambda x: lambda x, y: x + y

    node, source = parser.parse_entity(l, future_features=())
    expected_node_src = 'lambda x: (lambda x, y: (x + y))'
    self.assertAstMatches(node, source)
    self.assertAstMatches(node, expected_node_src)
    self.assertEqual(source, 'lambda x: lambda x, y: x + y')

    node, source = parser.parse_entity(l(0), future_features=())
    expected_node_src = 'lambda x, y: (x + y)'
    self.assertAstMatches(node, source)
    self.assertAstMatches(node, expected_node_src)
    self.assertEqual(source, 'lambda x, y: x + y')

  def test_parse_lambda_resolution_ambiguous(self):
    # [FUNCTION DOC]
    # Defines a lambda returning a lambda with the same parameter name on one line. Asserts that
    # parsing either the outer or the inner lambda raises an unsupported-language-element error
    # whose message mentions multiple definitions and lists both candidate lambdas.

    l = lambda x: lambda x: 2 * x

    expected_exception_text = re.compile(r'found multiple definitions'
                                         r'.+'
                                         r'\(?lambda x: \(?lambda x'
                                         r'.+'
                                         r'\(?lambda x: \(?2', re.DOTALL)

    with self.assertRaisesRegex(
        errors.UnsupportedLanguageElementError,
        expected_exception_text):
      parser.parse_entity(l, future_features=())

    with self.assertRaisesRegex(
        errors.UnsupportedLanguageElementError,
        expected_exception_text):
      parser.parse_entity(l(0), future_features=())

  def assertMatchesWithPotentialGarbage(self, source, expected, garbage):
    # [FUNCTION DOC]
    # HELPER: custom assertion used by the lambda tests.
    # Asserts that source (text) equals either expected or expected followed by garbage (extra
    # trailing context that runtimes without end-column tracking may include). Returns nothing.
    # In runtimes which don't track end_col_number, the source contains the
    # entire line, which in turn may have garbage from the surrounding context.
    self.assertIn(source, (expected, expected + garbage))

  def test_parse_lambda_multiline(self):
    # [FUNCTION DOC]
    # Defines a nested lambda whose body continues on a second line (with a trailing comment) in
    # parentheses. Parses the outer and inner lambdas; asserts trees match and the source texts
    # span both lines, allowing an optional trailing closing parenthesis.

    l = (
        lambda x: lambda y: x + y  # pylint:disable=g-long-lambda
        - 1)

    node, source = parser.parse_entity(l, future_features=())
    expected_node_src = 'lambda x: (lambda y: ((x + y) - 1))'
    self.assertAstMatches(node, expected_node_src)
    self.assertMatchesWithPotentialGarbage(
        source, ('lambda x: lambda y: x + y  # pylint:disable=g-long-lambda\n'
                 '        - 1'), ')')

    node, source = parser.parse_entity(l(0), future_features=())
    expected_node_src = 'lambda y: ((x + y) - 1)'
    self.assertAstMatches(node, expected_node_src)
    self.assertMatchesWithPotentialGarbage(
        source, ('lambda y: x + y  # pylint:disable=g-long-lambda\n'
                 '        - 1'), ')')

  def test_parse_lambda_in_expression(self):
    # [FUNCTION DOC]
    # Defines a tuple of two nested lambdas on adjacent lines. Parses each outer lambda and each
    # inner lambda and asserts the right tree and source text are found for each, allowing an
    # optional trailing comma in the source.

    l = (
        lambda x: lambda y: x + y + 1,
        lambda x: lambda y: x + y + 2,
        )

    node, source = parser.parse_entity(l[0], future_features=())
    expected_node_src = 'lambda x: (lambda y: ((x + y) + 1))'
    self.assertAstMatches(node, expected_node_src)
    self.assertMatchesWithPotentialGarbage(
        source, 'lambda x: lambda y: x + y + 1', ',')

    node, source = parser.parse_entity(l[0](0), future_features=())
    expected_node_src = 'lambda y: ((x + y) + 1)'
    self.assertAstMatches(node, expected_node_src)
    self.assertMatchesWithPotentialGarbage(
        source, 'lambda y: x + y + 1', ',')

    node, source = parser.parse_entity(l[1], future_features=())
    expected_node_src = 'lambda x: (lambda y: ((x + y) + 2))'
    self.assertAstMatches(node, expected_node_src)
    self.assertMatchesWithPotentialGarbage(source,
                                           'lambda x: lambda y: x + y + 2', ',')

    node, source = parser.parse_entity(l[1](0), future_features=())
    expected_node_src = 'lambda y: ((x + y) + 2)'
    self.assertAstMatches(node, expected_node_src)
    self.assertMatchesWithPotentialGarbage(source, 'lambda y: x + y + 2', ',')

  def test_parse_lambda_complex_body(self):
    # [FUNCTION DOC]
    # Parses a lambda whose body is a multi-line tuple containing a call, attribute access, a
    # slice, a string and a number. Asserts the tree matches and the source text covers all lines,
    # optionally with the closing parenthesis line when end positions are tracked.

    l = lambda x: (  # pylint:disable=g-long-lambda
        x.y(
            [],
            x.z,
            (),
            x[0:2],
        ),
        x.u,
        'abc',
        1,
    )

    node, source = parser.parse_entity(l, future_features=())
    expected_node_src = "lambda x: (x.y([], x.z, (), x[0:2]), x.u, 'abc', 1)"
    self.assertAstMatches(node, expected_node_src)

    base_source = ('lambda x: (  # pylint:disable=g-long-lambda\n'
                   '        x.y(\n'
                   '            [],\n'
                   '            x.z,\n'
                   '            (),\n'
                   '            x[0:2],\n'
                   '        ),\n'
                   '        x.u,\n'
                   '        \'abc\',\n'
                   '        1,')
    # The complete source includes the trailing parenthesis. But that is only
    # detected in runtimes which correctly track end_lineno for ASTs.
    self.assertMatchesWithPotentialGarbage(source, base_source, '\n    )')

  def test_parse_lambda_function_call_definition(self):
    # [FUNCTION DOC]
    # Passes a lambda as an argument of a call split across two lines, along with a keyword
    # argument, and asserts the lambda is parsed correctly with its source optionally followed by
    # the rest of the call text.

    def do_parse_and_test(lam, **unused_kwargs):
      # [FUNCTION DOC]
      # HELPER: receives a lambda (and ignored keyword arguments), parses it and asserts the
      # tree and source text are the expected identity lambda. Returns nothing.
      node, source = parser.parse_entity(lam, future_features=())
      expected_node_src = 'lambda x: x'
      self.assertAstMatches(node, expected_node_src)
      self.assertMatchesWithPotentialGarbage(
          source, 'lambda x: x', ', named_arg=1)')

    do_parse_and_test(  # Intentional line break
        lambda x: x, named_arg=1)

  def test_parse_entity_print_function(self):
    # [FUNCTION DOC]
    # Parses a function that prints its argument while requesting the print-function future
    # feature; asserts parsing succeeds and the node name is the function's name.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function to be parsed; prints its argument.
      print(x)

    node, _ = parser.parse_entity(f, future_features=('print_function',))
    self.assertEqual('f', node.name)

  def test_parse_comments(self):
    # [FUNCTION DOC]
    # Parses a function whose body starts with a comment; asserts parsing succeeds and the node
    # name is the function's name.

    def f():
      # [FUNCTION DOC]
      # HELPER: function to be parsed; does nothing.
      # unindented comment
      pass

    node, _ = parser.parse_entity(f, future_features=())
    self.assertEqual('f', node.name)

  def test_parse_multiline_strings(self):
    # [FUNCTION DOC]
    # Parses a function containing a multi-line string literal whose continuation lines have no
    # indentation; asserts parsing succeeds and the node name is the function's name.

    def f():
      # [FUNCTION DOC]
      # HELPER: function to be parsed; prints a multi-line string.
      print("""
multiline
string""")

    node, _ = parser.parse_entity(f, future_features=())
    self.assertEqual('f', node.name)

  def _eval_code(self, code, name):
    # [FUNCTION DOC]
    # HELPER: shared by the dedent tests.
    # Executes the code text in a fresh empty global namespace and returns the global object
    # called name (e.g. a function defined by the code). Side effect: runs the given code.
    globs = {}
    exec(code, globs)  # pylint:disable=exec-used
    return globs[name]

  def test_dedent_block_basic(self):
    # [FUNCTION DOC]
    # Dedents an indented function definition, evaluates it and asserts the function behaves
    # correctly (returns the negative absolute value for sample inputs 1 and -1).

    code = """
    def f(x):
      if x > 0:
        return -x
      return x
    """

    f = self._eval_code(parser.dedent_block(code), 'f')
    self.assertEqual(f(1), -1)
    self.assertEqual(f(-1), -1)

  def test_dedent_block_comments_out_of_line(self):
    # [FUNCTION DOC]
    # Dedents a function whose body is interleaved with comment-only lines at unrelated
    # indentation levels; evaluates it and asserts it still behaves correctly.

    code = """
  ###
    def f(x):
###
      if x > 0:
  ###
        return -x
          ###
  ###
      return x
      ###
    """

    f = self._eval_code(parser.dedent_block(code), 'f')
    self.assertEqual(f(1), -1)
    self.assertEqual(f(-1), -1)

  def test_dedent_block_multiline_string(self):
    # [FUNCTION DOC]
    # Dedents a function with a multi-line docstring and a multi-line string return value;
    # asserts string contents are not altered (the docstring keeps its original indentation,
    # except that on Python 3.13 and later it is compared after stripping because the compiler
    # removes docstring indentation).

    code = """
    def f():
      '''
      Docstring.
      '''
      return '''
  1
    2
      3'''
    """

    f = self._eval_code(parser.dedent_block(code), 'f')
    if sys.version_info >= (3, 13):
      self.assertEqual(f.__doc__.strip(), 'Docstring.')
    else:
      self.assertEqual(f.__doc__, '\n      Docstring.\n      ')

    self.assertEqual(f(), '\n  1\n    2\n      3')

  def test_dedent_block_multiline_expression(self):
    # [FUNCTION DOC]
    # Dedents a function returning a parenthesized tuple whose continuation lines have odd
    # indentation; evaluates it and asserts the tuple of 1, 2, 3 is returned.

    code = """
    def f():
      return (1,
2,
        3)
    """

    f = self._eval_code(parser.dedent_block(code), 'f')
    self.assertEqual(f(), (1, 2, 3))

  def test_dedent_block_continuation(self):
    # [FUNCTION DOC]
    # Dedents a function using a backslash line continuation in an assignment; evaluates it and
    # asserts it returns 1.

    code = r"""
    def f():
      a = \
          1
      return a
    """

    f = self._eval_code(parser.dedent_block(code), 'f')
    self.assertEqual(f(), 1)

  def test_dedent_block_continuation_in_string(self):
    # [FUNCTION DOC]
    # Dedents a function containing a string literal with a backslash continuation inside it;
    # evaluates it and asserts the result text is "a" followed by three spaces and "b", showing
    # the continuation removal keeps the following line's leading whitespace.

    code = r"""
    def f():
      a = "a \
  b"
      return a
    """

    f = self._eval_code(parser.dedent_block(code), 'f')
    self.assertEqual(f(), 'a   b')

  def test_parse_expression(self):
    # [FUNCTION DOC]
    # Parses the text of an attribute access expression and asserts the result is an attribute
    # node with the expected base name and attribute name.
    node = parser.parse_expression('a.b')
    self.assertEqual('a', node.value.id)
    self.assertEqual('b', node.attr)

  def test_unparse(self):
    # [FUNCTION DOC]
    # Hand-builds a conditional statement tree (if constant one: assign name b to a; else assign
    # a string to a), unparses it and asserts the text equals the expected code with the encoding
    # marker first and four-space indentation, ignoring surrounding whitespace.
    node = gast.If(
        test=gast.Constant(1, kind=None),
        body=[
            gast.Assign(
                targets=[
                    gast.Name(
                        'a',
                        ctx=gast.Store(),
                        annotation=None,
                        type_comment=None)
                ],
                value=gast.Name(
                    'b', ctx=gast.Load(), annotation=None, type_comment=None))
        ],
        orelse=[
            gast.Assign(
                targets=[
                    gast.Name(
                        'a',
                        ctx=gast.Store(),
                        annotation=None,
                        type_comment=None)
                ],
                value=gast.Constant('c', kind=None))
        ])

    source = parser.unparse(node, indentation='  ')
    self.assertEqual(
        textwrap.dedent("""
            # coding=utf-8
            if 1:
                a = b
            else:
                a = 'c'
        """).strip(), source.strip())

  def test_ext_slice_roundtrip(self):
    # [FUNCTION DOC]
    # Parses a function using multi-dimensional slices, unparses the tree back to text and
    # asserts the text re-parses to a matching tree (statement comparison, not expression).
    def ext_slice(n):
      # [FUNCTION DOC]
      # HELPER: function to be parsed; returns a tuple of three multi-dimensional slices of n.
      return n[:, :], n[0, :], n[:, 0]

    node, _ = parser.parse_entity(ext_slice, future_features=())
    source = parser.unparse(node)
    self.assertAstMatches(node, source, expr=False)

if __name__ == '__main__':
  test.main()
