# coding=utf-8
# [MODULE DOC]
# Unit tests for the autograph loader module (loading syntax trees and source text as modules).
# Covers a parse-then-load round trip, loading a hand-built tree, loading UTF-8 source with
# non-ASCII text, and tolerance of an already-deleted temporary file. Defines one test case class.
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
"""Tests for loader module."""

import os
import textwrap

import gast

from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


class LoaderTest(test.TestCase):
  # [CLASS DOC]
  # Test case for loader.load_ast and loader.load_source. Provides a custom AST-comparison
  # assertion and has no shared fixtures.

  def assertAstMatches(self, actual_node, expected_node_src):
    # [FUNCTION DOC]
    # HELPER: custom assertion for the loader tests.
    # Parses expected_node_src (source text) and takes its first top-level statement, then asserts
    # the actual syntax node structurally matches it. On failure the message shows pretty-printed
    # expected and actual trees. Returns nothing.
    expected_node = gast.parse(expected_node_src).body[0]

    msg = 'AST did not match expected:\n{}\nActual:\n{}'.format(
        pretty_printer.fmt(expected_node),
        pretty_printer.fmt(actual_node))
    self.assertTrue(ast_util.matches(actual_node, expected_node), msg)

  def test_parse_load_identity(self):
    # [FUNCTION DOC]
    # Parses a local function into a syntax tree, loads it as a module, and checks that both the
    # module's re-read source and the original function's dedented source match the tree, i.e.
    # parse followed by load is an identity on the code.

    def test_fn(x):
      # [FUNCTION DOC]
      # HELPER: function under test; returns an empty string or the argument plus one,
      # depending on a flag that is always true.
      a = True
      b = ''
      if a:
        b = (x + 1)
      return b

    node, _ = parser.parse_entity(test_fn, future_features=())
    module, _, _ = loader.load_ast(node)
    source = tf_inspect.getsource(module.test_fn)
    expected_node_src = textwrap.dedent(tf_inspect.getsource(test_fn))

    self.assertAstMatches(node, source)
    self.assertAstMatches(node, expected_node_src)

  def test_load_ast(self):
    # [FUNCTION DOC]
    # Hand-builds a function definition tree (one parameter, returns parameter plus one), loads it,
    # and checks the returned source and the file written to disk both match the tree and the
    # expected text, and that calling the loaded function with 1 returns 2.
    node = gast.FunctionDef(
        name='f',
        args=gast.arguments(
            args=[
                gast.Name(
                    'a', ctx=gast.Param(), annotation=None, type_comment=None)
            ],
            posonlyargs=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]),
        body=[
            gast.Return(
                gast.BinOp(
                    op=gast.Add(),
                    left=gast.Name(
                        'a',
                        ctx=gast.Load(),
                        annotation=None,
                        type_comment=None),
                    right=gast.Constant(1, kind=None)))
        ],
        decorator_list=[],
        returns=None,
        type_comment=None)

    module, source, _ = loader.load_ast(node)

    expected_node_src = """
      # coding=utf-8
      def f(a):
          return (a + 1)
    """
    expected_node_src = textwrap.dedent(expected_node_src)

    self.assertAstMatches(node, source)
    self.assertAstMatches(node, expected_node_src)

    self.assertEqual(2, module.f(1))
    with open(module.__file__, 'r') as temp_output:
      self.assertAstMatches(node, temp_output.read())

  def test_load_source(self):
    # [FUNCTION DOC]
    # Loads source text containing a function with a non-ASCII (Japanese and mathematical symbol)
    # docstring and checks the function works (1 gives 2) and its docstring is preserved exactly.
    test_source = textwrap.dedent(u"""
      # coding=utf-8
      def f(a):
        '日本語 Δθₜ ← Δθₜ₋₁ + ∇Q(sₜ, aₜ)(rₜ + γₜ₊₁ max Q(⋅))'
        return a + 1
    """)
    module, _ = loader.load_source(test_source, delete_on_exit=True)
    self.assertEqual(module.f(1), 2)
    self.assertEqual(
        module.f.__doc__, '日本語 Δθₜ ← Δθₜ₋₁ + ∇Q(sₜ, aₜ)(rₜ + γₜ₊₁ max Q(⋅))')

  def test_cleanup(self):
    # [FUNCTION DOC]
    # Loads empty source with delete-on-exit enabled, then deletes the temporary file by hand so
    # the later exit-time cleanup finds it missing. Asserts nothing directly; the scenario shows
    # the cleanup tolerates a vanished file.
    test_source = textwrap.dedent('')
    _, filename = loader.load_source(test_source, delete_on_exit=True)
    # Clean up the file before loader.py tries to remove it, to check that the
    # latter can deal with that situation.
    os.unlink(filename)

if __name__ == '__main__':
  test.main()
