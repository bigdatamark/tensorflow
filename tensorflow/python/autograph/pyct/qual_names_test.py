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
# Unit tests for the qual_names module: construction, text/simple-symbol/AST forms, equality,
# hashing, literals, support sets and ordering of QN objects, plus the QnResolver annotation of
# parsed code. Run as a test binary via the platform test main. No import-time side effects.
"""Tests for qual_names module."""

import textwrap

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct.qual_names import QN
from tensorflow.python.autograph.pyct.qual_names import resolve
from tensorflow.python.platform import test


class QNTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the QN class itself. No shared fixtures; each test builds its own QNs.

  def test_from_str(self):
    # [FUNCTION DOC]
    # Parses the strings for a simple, attribute and subscript name and asserts the results equal
    # QNs built by hand.
    a = QN('a')
    b = QN('b')
    a_dot_b = QN(a, attr='b')
    a_sub_b = QN(a, subscript=b)
    self.assertEqual(qual_names.from_str('a.b'), a_dot_b)
    self.assertEqual(qual_names.from_str('a'), a)
    self.assertEqual(qual_names.from_str('a[b]'), a_sub_b)

  def test_basic(self):
    # [FUNCTION DOC]
    # Checks a simple name and an attribute name: element tuple, text form, simple-symbol form,
    # AST fields, composite flag and parent. Asserts that asking a simple name for its parent
    # raises a value error.
    a = QN('a')
    self.assertEqual(a.qn, ('a',))
    self.assertEqual(str(a), 'a')
    self.assertEqual(a.ssf(), 'a')
    self.assertEqual(a.ast().id, 'a')
    self.assertFalse(a.is_composite())
    with self.assertRaises(ValueError):
      _ = a.parent

    a_b = QN(a, attr='b')
    self.assertEqual(a_b.qn, (a, 'b'))
    self.assertEqual(str(a_b), 'a.b')
    self.assertEqual(a_b.ssf(), 'a_b')
    self.assertEqual(a_b.ast().value.id, 'a')
    self.assertEqual(a_b.ast().attr, 'b')
    self.assertTrue(a_b.is_composite())
    self.assertEqual(a_b.parent.qn, ('a',))

  def test_subscripts(self):
    # [FUNCTION DOC]
    # Checks subscript names, including a nested subscript: tuple contents, text form,
    # simple-symbol form, AST value and slice fields, flags and parent. Asserts that building a
    # QN with a string base and a string positional attribute-like argument raises a value error.
    a = QN('a')
    b = QN('b')
    a_sub_b = QN(a, subscript=b)
    self.assertEqual(a_sub_b.qn, (a, b))
    self.assertEqual(str(a_sub_b), 'a[b]')
    self.assertEqual(a_sub_b.ssf(), 'a_sub_b')
    self.assertEqual(a_sub_b.ast().value.id, 'a')
    self.assertEqual(a_sub_b.ast().slice.id, 'b')
    self.assertTrue(a_sub_b.is_composite())
    self.assertTrue(a_sub_b.has_subscript())
    self.assertEqual(a_sub_b.parent.qn, ('a',))

    c = QN('c')
    b_sub_c = QN(b, subscript=c)
    a_sub_b_sub_c = QN(a, subscript=b_sub_c)
    self.assertEqual(a_sub_b_sub_c.qn, (a, b_sub_c))
    self.assertTrue(a_sub_b_sub_c.is_composite())
    self.assertTrue(a_sub_b_sub_c.has_subscript())
    self.assertEqual(b_sub_c.qn, (b, c))
    self.assertEqual(str(a_sub_b_sub_c), 'a[b[c]]')
    self.assertEqual(a_sub_b_sub_c.ssf(), 'a_sub_b_sub_c')
    self.assertEqual(a_sub_b_sub_c.ast().value.id, 'a')
    self.assertEqual(a_sub_b_sub_c.ast().slice.value.id, 'b')
    self.assertEqual(a_sub_b_sub_c.ast().slice.slice.id, 'c')
    self.assertEqual(b_sub_c.ast().slice.id, 'c')
    self.assertEqual(a_sub_b_sub_c.parent.qn, ('a',))
    with self.assertRaises(ValueError):
      QN('a', 'b')

  def test_equality(self):
    # [FUNCTION DOC]
    # Asserts that separately built but identical QNs (simple, attribute, subscript) compare equal
    # without being the same object, and that different forms are not equal.
    a = QN('a')
    a2 = QN('a')
    a_b = QN(a, attr='b')
    self.assertEqual(a2.qn, ('a',))
    with self.assertRaises(ValueError):
      _ = a.parent

    a_b2 = QN(a, attr='b')
    self.assertEqual(a_b2.qn, (a, 'b'))
    self.assertEqual(a_b2.parent.qn, ('a',))

    self.assertTrue(a2 == a)
    self.assertFalse(a2 is a)

    self.assertTrue(a_b.parent == a)
    self.assertTrue(a_b2.parent == a)

    self.assertTrue(a_b2 == a_b)
    self.assertFalse(a_b2 is a_b)
    self.assertFalse(a_b2 == a)
    a_sub_b = QN(a, subscript='b')
    a_sub_b2 = QN(a, subscript='b')
    self.assertTrue(a_sub_b == a_sub_b2)
    self.assertFalse(a_sub_b == a_b)

  def test_nested_attrs_subscripts(self):
    # [FUNCTION DOC]
    # Builds four mixed attribute/subscript nestings over a, b, c and asserts their text forms
    # and that they are all pairwise unequal.
    a = QN('a')
    b = QN('b')
    c = QN('c')
    b_sub_c = QN(b, subscript=c)
    a_sub_b_sub_c = QN(a, subscript=b_sub_c)

    b_dot_c = QN(b, attr='c')
    a_sub__b_dot_c = QN(a, subscript=b_dot_c)

    a_sub_b = QN(a, subscript=b)
    a_sub_b__dot_c = QN(a_sub_b, attr='c')

    a_dot_b = QN(a, attr='b')
    a_dot_b_sub_c = QN(a_dot_b, subscript=c)

    self.assertEqual(str(a_sub_b_sub_c), 'a[b[c]]')
    self.assertEqual(str(a_sub__b_dot_c), 'a[b.c]')
    self.assertEqual(str(a_sub_b__dot_c), 'a[b].c')
    self.assertEqual(str(a_dot_b_sub_c), 'a.b[c]')

    self.assertNotEqual(a_sub_b_sub_c, a_sub__b_dot_c)
    self.assertNotEqual(a_sub_b_sub_c, a_sub_b__dot_c)
    self.assertNotEqual(a_sub_b_sub_c, a_dot_b_sub_c)

    self.assertNotEqual(a_sub__b_dot_c, a_sub_b__dot_c)
    self.assertNotEqual(a_sub__b_dot_c, a_dot_b_sub_c)

    self.assertNotEqual(a_sub_b__dot_c, a_dot_b_sub_c)

  def test_hashable(self):
    # [FUNCTION DOC]
    # Uses QNs as mapping keys and asserts lookup by an equal QN works and a missing key is
    # absent.
    d = {QN('a'): 'a', QN('b'): 'b'}
    self.assertEqual(d[QN('a')], 'a')
    self.assertEqual(d[QN('b')], 'b')
    self.assertNotIn(QN('c'), d)

  def test_literals(self):
    # [FUNCTION DOC]
    # Asserts a subscript by string-literal differs in equality and hash from a subscript by
    # symbol of the same text, and that string and number literals give constant AST slices and
    # the expected text forms (quoted string, bare number).
    a = QN('a')
    a_sub_str_b = QN(a, subscript=QN(qual_names.Literal('b')))
    a_sub_b = QN(a, subscript=QN('b'))

    self.assertNotEqual(a_sub_str_b, a_sub_b)
    self.assertNotEqual(hash(a_sub_str_b), hash(a_sub_b))
    self.assertEqual(a_sub_str_b.ast().slice.value, 'b')
    self.assertEqual(str(a_sub_str_b), "a['b']")

    a_sub_three = QN(a, subscript=QN(qual_names.Literal(3)))
    self.assertEqual(a_sub_three.ast().slice.value, 3)
    self.assertEqual(str(a_sub_three), 'a[3]')

  def test_support_set(self):
    # [FUNCTION DOC]
    # Asserts the support set of simple, attribute and subscript names contains only the root
    # symbol(s) and any index symbol.
    a = QN('a')
    b = QN('b')
    c = QN('c')
    a_sub_b = QN(a, subscript=b)
    a_dot_b = QN(a, attr='b')
    a_dot_b_dot_c = QN(a_dot_b, attr='c')
    a_dot_b_sub_c = QN(a_dot_b, subscript=c)

    self.assertSetEqual(a.support_set, set((a,)))
    self.assertSetEqual(a_sub_b.support_set, set((a, b)))
    self.assertSetEqual(a_dot_b.support_set, set((a,)))
    self.assertSetEqual(a_dot_b_dot_c.support_set, set((a,)))
    self.assertSetEqual(a_dot_b_sub_c.support_set, set((a, c)))

  def test_comparison(self):
    # [FUNCTION DOC]
    # Asserts ordering of QNs follows the text form, in particular that a string literal (shown
    # with a leading quote) sorts relative to symbols by that quote character, using a character
    # just below the quote as a symbol name.
    less_than_apos = chr(ord('\'') - 1)

    self.assertGreater(QN('z'), QN(qual_names.Literal('a')))
    self.assertLess(QN(less_than_apos), QN(qual_names.Literal('a')))

    self.assertGreater(QN(qual_names.Literal('z')), QN(less_than_apos))
    self.assertLess(QN(qual_names.Literal('a')), QN('z'))


class QNResolverTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the resolve function and QnResolver: parse sample source, annotate, and check the
  # QN text attached to selected nodes.

  def assertQNStringIs(self, node, qn_str):
    # [FUNCTION DOC]
    # HELPER: custom assertion shared by the resolver tests.
    # Asserts the text form of the QN annotation on the node equals the expected string.
    self.assertEqual(str(anno.getanno(node, anno.Basic.QN)), qn_str)

  def test_resolve(self):
    # [FUNCTION DOC]
    # Parses several expression statements (names, attributes, tuples, lists, a call), resolves
    # them and asserts the annotated QN text of the names, attributes and call pieces.
    samples = """
      a
      a.b
      (c, d.e)
      [f, (g.h.i)]
      j(k, l)
    """
    nodes = parser.parse(textwrap.dedent(samples), single_node=False)
    nodes = tuple(resolve(node).value for node in nodes)

    self.assertQNStringIs(nodes[0], 'a')
    self.assertQNStringIs(nodes[1], 'a.b')
    self.assertQNStringIs(nodes[2].elts[0], 'c')
    self.assertQNStringIs(nodes[2].elts[1], 'd.e')
    self.assertQNStringIs(nodes[3].elts[0], 'f')
    self.assertQNStringIs(nodes[3].elts[1], 'g.h.i')
    self.assertQNStringIs(nodes[4].func, 'j')
    self.assertQNStringIs(nodes[4].args[0], 'k')
    self.assertQNStringIs(nodes[4].args[1], 'l')

  def test_subscript_resolve(self):
    # [FUNCTION DOC]
    # Parses expressions mixing attributes and nested subscripts, resolves them and asserts each
    # annotated QN text reproduces the original expression.
    samples = """
      x[i]
      x[i.b]
      a.b[c]
      a.b[x.y]
      a[z[c]]
      a[b[c[d]]]
      a[b].c
      a.b.c[d].e.f
      a.b[c[d]].e.f
      a.b[c[d.e.f].g].h
    """
    nodes = parser.parse(textwrap.dedent(samples), single_node=False)
    nodes = tuple(resolve(node).value for node in nodes)

    self.assertQNStringIs(nodes[0], 'x[i]')
    self.assertQNStringIs(nodes[1], 'x[i.b]')
    self.assertQNStringIs(nodes[2], 'a.b[c]')
    self.assertQNStringIs(nodes[3], 'a.b[x.y]')
    self.assertQNStringIs(nodes[4], 'a[z[c]]')
    self.assertQNStringIs(nodes[5], 'a[b[c[d]]]')
    self.assertQNStringIs(nodes[6], 'a[b].c')
    self.assertQNStringIs(nodes[7], 'a.b.c[d].e.f')
    self.assertQNStringIs(nodes[8], 'a.b[c[d]].e.f')
    self.assertQNStringIs(nodes[9], 'a.b[c[d.e.f].g].h')

  def test_function_calls(self):
    # [FUNCTION DOC]
    # Parses expressions involving calls combined with attributes and subscripts, resolves them
    # and asserts the QNs are attached to the callee or base parts, not to the call results.
    samples = """
      a.b
      a.b()
      a().b
      z[i]
      z[i]()
      z()[i]
    """
    nodes = parser.parse(textwrap.dedent(samples), single_node=False)
    nodes = tuple(resolve(node).value for node in nodes)
    self.assertQNStringIs(nodes[0], 'a.b')
    self.assertQNStringIs(nodes[1].func, 'a.b')
    self.assertQNStringIs(nodes[2].value.func, 'a')
    self.assertQNStringIs(nodes[3], 'z[i]')
    self.assertQNStringIs(nodes[4].func, 'z[i]')
    self.assertQNStringIs(nodes[5].value.func, 'z')


if __name__ == '__main__':
  test.main()
