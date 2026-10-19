# [MODULE DOC]
# Unit tests for the AutoGraph converter core module. Covers (1) round-tripping the conversion
# options through their AST form: the AST is embedded in generated code, reloaded with a fake
# generated-code namespace, and the resulting options are compared with the original; and
# (2) the base converter's directive lookup: single definition, default fallback, several
# consistent definitions, and several inconsistent definitions (which must raise a value error).
# Uses the converter testing base class for transformation helpers. Runs the TensorFlow test main
# when executed as a script.
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
"""Tests for converter module."""

import types

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.platform import test


class TestConverter(converter.Base):
  # [CLASS DOC]
  # HELPER: minimal concrete converter used only to exercise the directive lookup of the base
  # converter class in the tests below. Adds no behavior.
  pass


class ConversionOptionsTest(converter_testing.TestCase):
  # [CLASS DOC]
  # Test case for the conversion options class.

  def test_to_ast(self):
    # [FUNCTION DOC]
    # Scenario: default conversion options are converted to an AST expression, embedded in a
    # generated function that returns it, loaded as a module, and evaluated with a fake
    # generated-code namespace exposing the options class and feature enumeration. Asserts the
    # recursive flag, the internal-convert flag and the feature set equal the originals, and that
    # the original user-requested flag is false.
    opts = converter.ConversionOptions()
    opts_ast = opts.to_ast()

    template = '''
    def f():
      return opts_ast
    '''
    opts_packed = templates.replace(template, opts_ast=opts_ast)

    reparsed, _, _ = loader.load_ast(opts_packed)
    fake_ag = types.ModuleType('fake_ag')
    fake_ag.ConversionOptions = converter.ConversionOptions
    fake_ag.Feature = converter.Feature
    reparsed.ag__ = fake_ag

    reparsed_opts = reparsed.f()

    self.assertEqual(opts.recursive, reparsed_opts.recursive)
    self.assertEqual(opts.user_requested, False)
    self.assertEqual(
        opts.internal_convert_user_code,
        reparsed_opts.internal_convert_user_code)
    self.assertEqual(opts.optional_features, reparsed_opts.optional_features)


class ConverterBaseTest(converter_testing.TestCase):
  # [CLASS DOC]
  # Test case for the base converter's directive-definition lookup. Each test transforms a tiny
  # function with no converters to obtain its AST and context, attaches directive data to the
  # reaching definitions of a symbol, and queries it through the test converter.

  def test_get_definition_directive_basic(self):
    # [FUNCTION DOC]
    # Scenario: one definition carries two directive arguments. Asserts that asking for one named
    # argument returns its expression (identifier foo).

    directive_key = object

    def f():
      # [FUNCTION DOC]
      # Test subject: assigns one to a local and returns it.
      a = 1
      return a

    _, node, ctx = self.transform(f, (), include_ast=True)

    symbol_a = node.body[1].value
    defs, = anno.getanno(symbol_a, anno.Static.ORIG_DEFINITIONS)
    defs.directives[directive_key] = {
        'test_arg': parser.parse_expression('foo'),
        'other_arg': parser.parse_expression('bar'),
    }
    c = TestConverter(ctx)
    value = c.get_definition_directive(symbol_a, directive_key, 'test_arg',
                                       None)
    self.assertEqual(value.id, 'foo')

  def test_get_definition_directive_default(self):
    # [FUNCTION DOC]
    # Scenario: no directive is attached to the symbol's definition. Asserts the supplied default
    # expression (identifier default) is returned.

    directive_key = object

    def f():
      # [FUNCTION DOC]
      # Test subject: assigns one to a local and returns it.
      a = 1
      return a

    _, node, ctx = self.transform(f, (), include_ast=True)

    symbol_a = node.body[1].value
    c = TestConverter(ctx)
    value = c.get_definition_directive(symbol_a, directive_key, 'test_arg',
                                       parser.parse_expression('default'))
    self.assertEqual(value.id, 'default')

  def test_get_definition_directive_multiple_consistent(self):
    # [FUNCTION DOC]
    # Scenario: a symbol reached by two definitions, both specifying the same value (foo) for the
    # queried argument but differing in another argument. Asserts the lookup returns foo.

    directive_key = object

    def f():
      # [FUNCTION DOC]
      # Test subject: assigns a local twice, once conditionally, and returns it.
      a = 1
      if a:
        a = 2
      return a

    _, node, ctx = self.transform(f, (), include_ast=True)

    symbol_a = node.body[2].value
    defs = anno.getanno(symbol_a, anno.Static.ORIG_DEFINITIONS)
    defs[0].directives[directive_key] = {
        'test_arg': parser.parse_expression('foo'),
        'other_arg': parser.parse_expression('bar'),
    }
    defs[1].directives[directive_key] = {
        'test_arg': parser.parse_expression('foo'),
        'other_arg': parser.parse_expression('baz'),
    }
    c = TestConverter(ctx)
    value = c.get_definition_directive(symbol_a, directive_key, 'test_arg',
                                       None)
    self.assertEqual(value.id, 'foo')

  def test_get_definition_directive_multiple_inconsistent(self):
    # [FUNCTION DOC]
    # Scenario: two reaching definitions give different values (foo and bar) for the queried
    # argument. Asserts the lookup raises a value error.

    directive_key = object

    def f():
      # [FUNCTION DOC]
      # Test subject: assigns a local twice, once conditionally, and returns it.
      a = 1
      if a:
        a = 2
      return a

    _, node, ctx = self.transform(f, (), include_ast=True)

    symbol_a = node.body[2].value
    defs = anno.getanno(symbol_a, anno.Static.ORIG_DEFINITIONS)
    defs[0].directives[directive_key] = {
        'test_arg': parser.parse_expression('foo'),
    }
    defs[1].directives[directive_key] = {
        'test_arg': parser.parse_expression('bar'),
    }
    c = TestConverter(ctx)
    with self.assertRaises(ValueError):
      c.get_definition_directive(symbol_a, directive_key, 'test_arg', None)


if __name__ == '__main__':
  test.main()
