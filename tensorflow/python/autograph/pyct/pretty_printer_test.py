# [MODULE DOC]
# Smoke tests for the autograph pretty_printer module: formatting a parsed tree containing
# byte and text literals, and a hand-built function tree. Only checks that formatting returns a
# non-null result (colored output is hard to inspect). Defines one test case class.
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
"""Tests for pretty_printer module."""

import ast
import textwrap

from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.platform import test


class PrettyPrinterTest(test.TestCase):
  # [CLASS DOC]
  # Test case for pretty_printer.fmt; stateless smoke tests with no shared fixtures.

  def test_unicode_bytes(self):
    # [FUNCTION DOC]
    # Parses a function returning a byte string, a text string and an ambiguous string literal
    # with the standard parser and asserts formatting the tree yields a non-null result.
    source = textwrap.dedent('''
    def f():
      return b'b', u'u', 'depends_py2_py3'
    ''')
    node = ast.parse(source)
    self.assertIsNotNone(pretty_printer.fmt(node))

  def test_format(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: builds the tree with legacy node types and fields (a parameter context, a
    # number node) that newer Python versions removed or deprecated, so it may fail or warn there.
    # Hand-builds a function definition tree (one parameter, returns it plus one) and asserts
    # formatting it yields a non-null result; no output content is checked.
    node = ast.FunctionDef(
        name='f',
        args=ast.arguments(
            args=[ast.Name(id='a', ctx=ast.Param())],
            vararg=None,
            kwarg=None,
            defaults=[]),
        body=[
            ast.Return(
                ast.BinOp(
                    op=ast.Add(),
                    left=ast.Name(id='a', ctx=ast.Load()),
                    right=ast.Num(1)))
        ],
        decorator_list=[],
        returns=None)
    # Just checking for functionality, the color control characters make it
    # difficult to inspect the result.
    self.assertIsNotNone(pretty_printer.fmt(node))


if __name__ == '__main__':
  test.main()
