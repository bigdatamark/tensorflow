# [MODULE DOC]
# Unit tests for the ast_util module: symbol renaming (names, attributes, nonlocal, global,
# annotations, function names), clean copying, keyword-to-dictionary conversion, pattern
# matching with wildcards, single-assignment decomposition and parallel tree walking.
#
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
"""Tests for ast_util module."""

import ast
import collections
import textwrap

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.platform import test


class AstUtilTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the syntax-tree utilities. Shared fixture: a per-test counter mapping
  # (target text, source text) pairs to invocation counts, used by the assignment tests. Also
  # provides custom assertion helpers for matching trees and patterns.

  def assertAstMatches(self, actual_node, expected_node_src):
    # [FUNCTION DOC]
    # HELPER: custom assertion used by the renaming tests.
    # Parses expected_node_src (text) wrapped in parentheses to get an expected expression node,
    # and asserts that the pattern matcher accepts actual_node against it; on failure the message
    # shows pretty-printed expected and actual trees.
    expected_node = gast.parse('({})'.format(expected_node_src)).body[0]
    msg = 'AST did not match expected:\n{}\nActual:\n{}'.format(
        pretty_printer.fmt(expected_node),
        pretty_printer.fmt(actual_node))
    self.assertTrue(ast_util.matches(actual_node, expected_node), msg)

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the base set-up, then creates the invocation-count mapping (default zero) used by the
    # assignment tests.
    super(AstUtilTest, self).setUp()
    self._invocation_counts = collections.defaultdict(lambda: 0)

  def test_rename_symbols_basic(self):
    # [FUNCTION DOC]
    # Parses a sum of two names, resolves qualified names, renames the first name, and checks the
    # renamed identifier is plain text and that the tree matches both its own unparsed source and
    # the expected source with the renamed first operand.
    node = parser.parse('a + b')
    node = qual_names.resolve(node)

    node = ast_util.rename_symbols(
        node, {qual_names.QN('a'): qual_names.QN('renamed_a')})
    source = parser.unparse(node, include_encoding_marker=False)
    expected_node_src = 'renamed_a + b'

    self.assertIsInstance(node.value.left.id, str)
    self.assertAstMatches(node, source)
    self.assertAstMatches(node, expected_node_src)

  def test_rename_symbols_attributes(self):
    # [FUNCTION DOC]
    # Renames the attribute chain b.c to a simple name in an assignment whose both sides use it
    # (the right side is b.c.d); expects the unparsed source to be the new name assigned from the
    # new name followed by d.
    node = parser.parse('b.c = b.c.d')
    node = qual_names.resolve(node)

    node = ast_util.rename_symbols(
        node, {qual_names.from_str('b.c'): qual_names.QN('renamed_b_c')})

    source = parser.unparse(node, include_encoding_marker=False)
    self.assertEqual(source.strip(), 'renamed_b_c = renamed_b_c.d')

  def test_rename_symbols_nonlocal(self):
    # [FUNCTION DOC]
    # Renames b in a nonlocal statement listing a, b, c; expects only b to change in the output.
    node = parser.parse('nonlocal a, b, c')
    node = qual_names.resolve(node)

    node = ast_util.rename_symbols(
        node, {qual_names.from_str('b'): qual_names.QN('renamed_b')})

    source = parser.unparse(node, include_encoding_marker=False)
    self.assertEqual(source.strip(), 'nonlocal a, renamed_b, c')

  def test_rename_symbols_global(self):
    # [FUNCTION DOC]
    # Renames b in a global statement listing a, b, c; expects only b to change in the output.
    node = parser.parse('global a, b, c')
    node = qual_names.resolve(node)

    node = ast_util.rename_symbols(
        node, {qual_names.from_str('b'): qual_names.QN('renamed_b')})

    source = parser.unparse(node, include_encoding_marker=False)
    self.assertEqual(source.strip(), 'global a, renamed_b, c')

  def test_rename_symbols_annotations(self):
    # [FUNCTION DOC]
    # Annotates the root node of a subscript expression, renames a symbol inside it, and checks the
    # custom annotation on the root is still the very same object afterwards.
    node = parser.parse('a[i]')
    node = qual_names.resolve(node)
    anno.setanno(node, 'foo', 'bar')
    orig_anno = anno.getanno(node, 'foo')

    node = ast_util.rename_symbols(node,
                                   {qual_names.QN('a'): qual_names.QN('b')})

    self.assertIs(anno.getanno(node, 'foo'), orig_anno)

  def test_rename_symbols_function(self):
    # [FUNCTION DOC]
    # Renames a function definition named f to f1 (no qualified-name resolution is run) and checks
    # the unparsed source shows the new name with the same body.
    node = parser.parse('def f():\n  pass')
    node = ast_util.rename_symbols(node,
                                   {qual_names.QN('f'): qual_names.QN('f1')})

    source = parser.unparse(node, include_encoding_marker=False)
    self.assertEqual(source.strip(), 'def f1():\n    pass')

  def test_copy_clean(self):
    # [FUNCTION DOC]
    # Sets a double-underscore-prefixed attribute on a parsed function node, copies it with
    # copy_clean, and checks the copy is a different object lacking that attribute.
    node = parser.parse(
        textwrap.dedent("""
      def f(a):
        return a + 1
    """))
    setattr(node, '__foo', 'bar')
    new_node = ast_util.copy_clean(node)
    self.assertIsNot(new_node, node)
    self.assertFalse(hasattr(new_node, '__foo'))

  def test_copy_clean_preserves_annotations(self):
    # [FUNCTION DOC]
    # Annotates a function node with two keys, copies it preserving only one key, and checks the
    # preserved annotation has its value in the copy while the other is absent.
    node = parser.parse(
        textwrap.dedent("""
      def f(a):
        return a + 1
    """))
    anno.setanno(node, 'foo', 'bar')
    anno.setanno(node, 'baz', 1)
    new_node = ast_util.copy_clean(node, preserve_annos={'foo'})
    self.assertEqual(anno.getanno(new_node, 'foo'), 'bar')
    self.assertFalse(anno.hasanno(new_node, 'baz'))

  def test_keywords_to_dict(self):
    # [FUNCTION DOC]
    # Parses a call with three keyword arguments, converts them to a dictionary node, appends a
    # return of that dictionary to a parsed one-parameter function, loads the tree as a module and
    # calls the function with 3. Expects the dictionary mapping the three argument names to 3, 1
    # and the text e.
    keywords = parser.parse_expression('f(a=b, c=1, d=\'e\')').keywords
    d = ast_util.keywords_to_dict(keywords)
    # Make sure we generate a usable dict node by attaching it to a variable and
    # compiling everything.
    node = parser.parse('def f(b): pass')
    node.body.append(ast.Return(d))
    result, _, _ = loader.load_ast(node)
    self.assertDictEqual(result.f(3), {'a': 3, 'c': 1, 'd': 'e'})

  def assertMatch(self, target_str, pattern_str):
    # [FUNCTION DOC]
    # HELPER: custom assertion for the matching tests.
    # Parses both texts as expressions and asserts the target matches the pattern.
    node = parser.parse_expression(target_str)
    pattern = parser.parse_expression(pattern_str)
    self.assertTrue(ast_util.matches(node, pattern))

  def assertNoMatch(self, target_str, pattern_str):
    # [FUNCTION DOC]
    # HELPER: custom assertion for the matching tests.
    # Parses both texts as expressions and asserts the target does not match the pattern.
    node = parser.parse_expression(target_str)
    pattern = parser.parse_expression(pattern_str)
    self.assertFalse(ast_util.matches(node, pattern))

  def test_matches_symbols(self):
    # [FUNCTION DOC]
    # Checks wildcard matching on names and binary operations: a name matches a wildcard but a
    # call does not; a sum matches a pattern with the same left operand and wildcard right
    # operand, but not with a different left operand or a different operator.
    self.assertMatch('foo', '_')
    self.assertNoMatch('foo()', '_')
    self.assertMatch('foo + bar', 'foo + _')
    self.assertNoMatch('bar + bar', 'foo + _')
    self.assertNoMatch('foo - bar', 'foo + _')

  def test_matches_function_args(self):
    # [FUNCTION DOC]
    # Checks matching of super-initializer calls: a wildcard argument list matches any argument
    # list (including none); a different method name does not match; explicit argument patterns
    # match only when the non-wildcard arguments are equal.
    self.assertMatch('super(Foo, self).__init__(arg1, arg2)',
                     'super(_).__init__(_)')
    self.assertMatch('super().__init__()', 'super(_).__init__(_)')
    self.assertNoMatch('super(Foo, self).bar(arg1, arg2)',
                       'super(_).__init__(_)')
    self.assertMatch('super(Foo, self).__init__()', 'super(Foo, _).__init__(_)')
    self.assertNoMatch('super(Foo, self).__init__()',
                       'super(Bar, _).__init__(_)')

  def _mock_apply_fn(self, target, source):
    # [FUNCTION DOC]
    # HELPER: callback passed to the single-assignment decomposition in the tests.
    # Unparses the target and source nodes to trimmed text and increments the invocation count for
    # that pair. Returns nothing.
    target = parser.unparse(target, include_encoding_marker=False)
    source = parser.unparse(source, include_encoding_marker=False)
    self._invocation_counts[(target.strip(), source.strip())] += 1

  def test_apply_to_single_assignments_dynamic_unpack(self):
    # [FUNCTION DOC]
    # Decomposes an assignment of one value to a three-name tuple; expects the callback to be called
    # once per name, paired with the value subscripted at positions zero, one and two.
    node = parser.parse('a, b, c = d')
    ast_util.apply_to_single_assignments(node.targets, node.value,
                                         self._mock_apply_fn)
    self.assertDictEqual(self._invocation_counts, {
        ('a', 'd[0]'): 1,
        ('b', 'd[1]'): 1,
        ('c', 'd[2]'): 1,
    })

  def test_apply_to_single_assignments_static_unpack(self):
    # [FUNCTION DOC]
    # Decomposes an assignment of a three-element tuple to a three-name tuple; expects the callback
    # once per name, paired with the corresponding element name.
    node = parser.parse('a, b, c = d, e, f')
    ast_util.apply_to_single_assignments(node.targets, node.value,
                                         self._mock_apply_fn)
    self.assertDictEqual(self._invocation_counts, {
        ('a', 'd'): 1,
        ('b', 'e'): 1,
        ('c', 'f'): 1,
    })

  def test_parallel_walk(self):
    # [FUNCTION DOC]
    # Walks a parsed function with itself in parallel and asserts each yielded pair is equal.
    src = """
      def f(a):
        return a + 1
    """
    node = parser.parse(textwrap.dedent(src))
    for child_a, child_b in ast_util.parallel_walk(node, node):
      self.assertEqual(child_a, child_b)

  def test_parallel_walk_string_leaves(self):
    # [FUNCTION DOC]
    # Walks a function containing a global statement (whose names are text leaves) with itself and
    # asserts each yielded pair is equal, showing text leaves are handled.
    src = """
      def f(a):
        global g
    """
    node = parser.parse(textwrap.dedent(src))
    for child_a, child_b in ast_util.parallel_walk(node, node):
      self.assertEqual(child_a, child_b)

  def test_parallel_walk_inconsistent_trees(self):
    # [FUNCTION DOC]
    # Walks a function against one with a structurally different expression and expects a value
    # error; then against one differing only in a constant's value and also expects a value error
    # (noted in a TODO as arguably too strict).
    node_1 = parser.parse(
        textwrap.dedent("""
      def f(a):
        return a + 1
    """))
    node_2 = parser.parse(
        textwrap.dedent("""
      def f(a):
        return a + (a * 2)
    """))
    node_3 = parser.parse(
        textwrap.dedent("""
      def f(a):
        return a + 2
    """))
    with self.assertRaises(ValueError):
      for _ in ast_util.parallel_walk(node_1, node_2):
        pass
    # There is not particular reason to reject trees that differ only in the
    # value of a constant.
    # TODO(mdan): This should probably be allowed.
    with self.assertRaises(ValueError):
      for _ in ast_util.parallel_walk(node_1, node_3):
        pass

  def assertLambdaNodes(self, matching_nodes, expected_bodies):
    # [FUNCTION DOC]
    # HELPER: custom assertion (not used by any test in this file).
    # Asserts the list matching_nodes has the same length as expected_bodies, that every item is a
    # lambda node, and that each lambda's unparsed, trimmed body text is among expected_bodies.
    self.assertEqual(len(matching_nodes), len(expected_bodies))
    for node in matching_nodes:
      self.assertIsInstance(node, gast.Lambda)
      self.assertIn(
          parser.unparse(node.body, include_encoding_marker=False).strip(),
          expected_bodies)


if __name__ == '__main__':
  test.main()
