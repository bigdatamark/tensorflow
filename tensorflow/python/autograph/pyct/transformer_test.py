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
# Unit tests for the transformer module (Base transformer and CodeGenerator): scoped state
# tracking, block visiting with post-processing, error robustness, origin info propagation to
# new and moved nodes, and basic text code generation. Each test defines small transformer
# subclasses inline and parses a local function. Run as a test binary; no import-time side
# effects. (The original docstring below wrongly says it tests the templates module.)
"""Tests for templates module."""

import re
import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.platform import test


class TransformerTest(test.TestCase):
  # [CLASS DOC]
  # Tests for transformer.Base. Shared fixtures: a method building a minimal Context and two
  # assertion helpers comparing annotations by identity.

  def _simple_context(self):
    # [FUNCTION DOC]
    # HELPER: test fixture. Builds a Context around an entity info named Test_fn with no source
    # code, file or namespace, no future features, and no namer or user context. Returns it.
    entity_info = transformer.EntityInfo(
        name='Test_fn',
        source_code=None,
        source_file=None,
        future_features=(),
        namespace=None)
    return transformer.Context(entity_info, None, None)

  def assertSameAnno(self, first, second, key):
    # [FUNCTION DOC]
    # HELPER: custom assertion. Asserts the annotation under key on the two nodes is the very
    # same object (identity).
    self.assertIs(anno.getanno(first, key), anno.getanno(second, key))

  def assertDifferentAnno(self, first, second, key):
    # [FUNCTION DOC]
    # HELPER: custom assertion. Asserts the annotation under key on the two nodes are different
    # objects (identity).
    self.assertIsNot(anno.getanno(first, key), anno.getanno(second, key))

  def test_state_tracking(self):
    # [FUNCTION DOC]
    # Scenario: a transformer records, on every node, the innermost loop state and condition
    # state objects, entering new ones when visiting while loops and if statements. It is run on
    # a function with nested and sibling loops/ifs. Asserts that nodes inside the same scope share
    # the state object, nodes in different scopes (nested or sibling) get different objects, and
    # states of the other kind are unaffected.

    class LoopState(object):
      # [CLASS DOC]
      # Empty state type tracked per while loop.
      pass

    class CondState(object):
      # [CLASS DOC]
      # Empty state type tracked per if statement.
      pass

    class TestTransformer(transformer.Base):
      # [CLASS DOC]
      # Transformer that stamps loop and condition state annotations on each node and manages
      # state scopes on while and if nodes.

      def visit(self, node):
        # [FUNCTION DOC]
        # Annotates the node with the current top loop state and condition state, then delegates
        # to the base visit and returns its result.
        anno.setanno(node, 'loop_state', self.state[LoopState].value)
        anno.setanno(node, 'cond_state', self.state[CondState].value)
        return super(TestTransformer, self).visit(node)

      def visit_While(self, node):
        # [FUNCTION DOC]
        # Enters a new loop state, visits children, exits it, and returns the node.
        self.state[LoopState].enter()
        node = self.generic_visit(node)
        self.state[LoopState].exit()
        return node

      def visit_If(self, node):
        # [FUNCTION DOC]
        # Enters a new condition state, visits children, exits it, and returns the node.
        self.state[CondState].enter()
        node = self.generic_visit(node)
        self.state[CondState].exit()
        return node

    tr = TestTransformer(self._simple_context())

    def test_function(a):
      # [FUNCTION DOC]
      # HELPER: sample input function that is parsed, never called; it has nested and sibling
      # while loops and if statements.
      a = 1
      while a:
        _ = 'a'
        if a > 2:
          _ = 'b'
          while True:
            raise '1'
        if a > 3:
          _ = 'c'
          while True:
            raise '1'

    node, _ = parser.parse_entity(test_function, future_features=())
    node = tr.visit(node)

    fn_body = node.body
    outer_while_body = fn_body[1].body
    self.assertSameAnno(fn_body[0], outer_while_body[0], 'cond_state')
    self.assertDifferentAnno(fn_body[0], outer_while_body[0], 'loop_state')

    first_if_body = outer_while_body[1].body
    self.assertDifferentAnno(outer_while_body[0], first_if_body[0],
                             'cond_state')
    self.assertSameAnno(outer_while_body[0], first_if_body[0], 'loop_state')

    first_inner_while_body = first_if_body[1].body
    self.assertSameAnno(first_if_body[0], first_inner_while_body[0],
                        'cond_state')
    self.assertDifferentAnno(first_if_body[0], first_inner_while_body[0],
                             'loop_state')

    second_if_body = outer_while_body[2].body
    self.assertDifferentAnno(first_if_body[0], second_if_body[0], 'cond_state')
    self.assertSameAnno(first_if_body[0], second_if_body[0], 'loop_state')

    second_inner_while_body = second_if_body[1].body
    self.assertDifferentAnno(first_inner_while_body[0],
                             second_inner_while_body[0], 'cond_state')
    self.assertDifferentAnno(first_inner_while_body[0],
                             second_inner_while_body[0], 'loop_state')

  def test_state_tracking_context_manager(self):
    # [FUNCTION DOC]
    # Scenario: like the state tracking test but the scope is entered with a with-statement on
    # the state stack while visiting if nodes. Asserts the condition state differs between
    # outside and inside an if, is the same for siblings inside it, and differs for a nested if.

    class CondState(object):
      # [CLASS DOC]
      # Empty state type tracked per if statement.
      pass

    class TestTransformer(transformer.Base):
      # [CLASS DOC]
      # Transformer that stamps the condition state on each node and scopes it with a context
      # manager when visiting if nodes.

      def visit(self, node):
        # [FUNCTION DOC]
        # Annotates the node with the current condition state, then delegates to the base visit.
        anno.setanno(node, 'cond_state', self.state[CondState].value)
        return super(TestTransformer, self).visit(node)

      def visit_If(self, node):
        # [FUNCTION DOC]
        # Visits children inside a scoped condition state and returns the visited node.
        with self.state[CondState]:
          return self.generic_visit(node)

    tr = TestTransformer(self._simple_context())

    def test_function(a):
      # [FUNCTION DOC]
      # HELPER: sample input function that is parsed, never called; it has nested ifs.
      a = 1
      if a > 2:
        _ = 'b'
        if a < 5:
          _ = 'c'
        _ = 'd'

    node, _ = parser.parse_entity(test_function, future_features=())
    node = tr.visit(node)

    fn_body = node.body
    outer_if_body = fn_body[1].body
    self.assertDifferentAnno(fn_body[0], outer_if_body[0], 'cond_state')
    self.assertSameAnno(outer_if_body[0], outer_if_body[2], 'cond_state')

    inner_if_body = outer_if_body[1].body
    self.assertDifferentAnno(inner_if_body[0], outer_if_body[0], 'cond_state')

  def test_visit_block_postprocessing(self):
    # [FUNCTION DOC]
    # Scenario: a transformer visits a function body with a post-processing callback that, upon
    # an assignment whose value is the name y, wraps it in a new if statement and redirects all
    # following statements into that if. Asserts the body has two statements: the first assign,
    # then an if whose body contains the moved assignment and the return.

    class TestTransformer(transformer.Base):
      # [CLASS DOC]
      # Transformer that rewrites function bodies using the block visitor and a callback.

      def _process_body_item(self, node):
        # [FUNCTION DOC]
        # HELPER: post-visit callback for the block visitor. For an assignment of the name y,
        # returns a new if statement (on x) containing the node together with that if's body as
        # the new destination. For any other node returns the node and no destination.
        if isinstance(node, gast.Assign) and (node.value.id == 'y'):
          if_node = gast.If(
              gast.Name(
                  'x', ctx=gast.Load(), annotation=None, type_comment=None),
              [node], [])
          return if_node, if_node.body
        return node, None

      def visit_FunctionDef(self, node):
        # [FUNCTION DOC]
        # Replaces the function body with the result of visiting the block with the
        # post-processing callback. Returns the function node.
        node.body = self.visit_block(
            node.body, after_visit=self._process_body_item)
        return node

    def test_function(x, y):
      # [FUNCTION DOC]
      # HELPER: sample input function that is parsed, never called.
      z = x
      z = y
      return z

    tr = TestTransformer(self._simple_context())

    node, _ = parser.parse_entity(test_function, future_features=())
    node = tr.visit(node)

    self.assertEqual(len(node.body), 2)
    self.assertIsInstance(node.body[0], gast.Assign)
    self.assertIsInstance(node.body[1], gast.If)
    self.assertIsInstance(node.body[1].body[0], gast.Assign)
    self.assertIsInstance(node.body[1].body[1], gast.Return)

  def test_robust_error_on_list_visit(self):
    # [FUNCTION DOC]
    # Scenario: a transformer wrongly calls the single-node visit with a list (an if body).
    # Asserts a value error is raised whose message says a tree node was expected and a list was
    # received.

    class BrokenTransformer(transformer.Base):
      # [CLASS DOC]
      # Deliberately buggy transformer that visits a list instead of a node.

      def visit_If(self, node):
        # [FUNCTION DOC]
        # Calls the single-node visit with the list body of the if (the bug under test), then
        # returns the node; never reaches the return because that call raises.
        # This is broken because visit expects a single node, not a list, and
        # the body of an if is a list.
        # Importantly, the default error handling in visit also expects a single
        # node.  Therefore, mistakes like this need to trigger a type error
        # before the visit called here installs its error handler.
        # That type error can then be caught by the enclosing call to visit,
        # and correctly blame the If node.
        self.visit(node.body)
        return node

    def test_function(x):
      # [FUNCTION DOC]
      # HELPER: sample input function that is parsed, never called.
      if x > 0:
        return x

    tr = BrokenTransformer(self._simple_context())

    node, _ = parser.parse_entity(test_function, future_features=())
    with self.assertRaises(ValueError) as cm:
      node = tr.visit(node)
    obtained_message = str(cm.exception)
    expected_message = r'expected "ast.AST", got "\<(type|class) \'list\'\>"'
    self.assertRegex(obtained_message, expected_message)

  def test_robust_error_on_ast_corruption(self):
    # [FUNCTION DOC]
    # Scenario: a transformer replaces an if body with a non-node object and then raises a value
    # error with a known message. Asserts the error that surfaces from visiting is that original
    # error (message contains the known text), i.e. the base error handling does not mask it.
    # A child class should not be able to be so broken that it causes the error
    # handling in `transformer.Base` to raise an exception.  Why not?  Because
    # then the original error location is dropped, and an error handler higher
    # up in the call stack gives misleading information.

    # Here we test that the error handling in `visit` completes, and blames the
    # correct original exception, even if the AST gets corrupted.

    class NotANode(object):
      # [CLASS DOC]
      # Empty class whose instance stands in for an invalid tree node.
      pass

    class BrokenTransformer(transformer.Base):
      # [CLASS DOC]
      # Deliberately buggy transformer that corrupts the tree and then fails.

      def visit_If(self, node):
        # [FUNCTION DOC]
        # Sets the if body to a non-node object, then raises a value error with a fixed message.
        node.body = NotANode()
        raise ValueError('I blew up')

    def test_function(x):
      # [FUNCTION DOC]
      # HELPER: sample input function that is parsed, never called.
      if x > 0:
        return x

    tr = BrokenTransformer(self._simple_context())

    node, _ = parser.parse_entity(test_function, future_features=())
    with self.assertRaises(ValueError) as cm:
      node = tr.visit(node)
    obtained_message = str(cm.exception)
    # The message should reference the exception actually raised, not anything
    # from the exception handler.
    expected_substring = 'I blew up'
    self.assertIn(expected_substring, obtained_message)

  def test_origin_info_propagated_to_new_nodes(self):
    # [FUNCTION DOC]
    # Scenario: a transformer replaces an if statement with a new pass statement. After
    # resolving origin info (file name test_file, starting line 100), asserts the new node
    # carries the origin of the replaced if statement (line 102).

    class TestTransformer(transformer.Base):
      # [CLASS DOC]
      # Transformer replacing every if statement with a pass statement.

      def visit_If(self, node):
        # [FUNCTION DOC]
        # Returns a new pass node in place of the if node.
        return gast.Pass()

    tr = TestTransformer(self._simple_context())

    def test_fn():
      # [FUNCTION DOC]
      # HELPER: sample input function that is parsed, never called.
      x = 1
      if x > 0:
        x = 1
      return x

    node, source = parser.parse_entity(test_fn, future_features=())
    origin_info.resolve(node, source, 'test_file', 100, 0)
    node = tr.visit(node)

    created_pass_node = node.body[1]
    # Takes the line number of the if statement.
    self.assertEqual(
        anno.getanno(created_pass_node, anno.Basic.ORIGIN).loc.lineno, 102)

  def test_origin_info_preserved_in_moved_nodes(self):
    # [FUNCTION DOC]
    # Scenario: a transformer replaces an if statement with the list of statements in its body.
    # After resolving origin info (starting line 100), asserts the moved statements keep their
    # own original lines (103 and 104) rather than the line of the if.

    class TestTransformer(transformer.Base):
      # [CLASS DOC]
      # Transformer replacing every if statement with the statements of its body.

      def visit_If(self, node):
        # [FUNCTION DOC]
        # Returns the body list of the if in place of the if node.
        return node.body

    tr = TestTransformer(self._simple_context())

    def test_fn():
      # [FUNCTION DOC]
      # HELPER: sample input function that is parsed, never called.
      x = 1
      if x > 0:
        x = 1
        x += 3
      return x

    node, source = parser.parse_entity(test_fn, future_features=())
    origin_info.resolve(node, source, 'test_file', 100, 0)
    node = tr.visit(node)

    assign_node = node.body[1]
    aug_assign_node = node.body[2]
    # Keep their original line numbers.
    self.assertEqual(
        anno.getanno(assign_node, anno.Basic.ORIGIN).loc.lineno, 103)
    self.assertEqual(
        anno.getanno(aug_assign_node, anno.Basic.ORIGIN).loc.lineno, 104)


class CodeGeneratorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for transformer.CodeGenerator, using a minimal Context fixture.

  def _simple_context(self):
    # [FUNCTION DOC]
    # HELPER: test fixture. Builds a Context around an entity info named test_fn with no source,
    # file or namespace, no future features, and no namer or user context. Returns it.
    entity_info = transformer.EntityInfo(
        name='test_fn',
        source_code=None,
        source_file=None,
        future_features=(),
        namespace=None)
    return transformer.Context(entity_info, None, None)

  def test_basic_codegen(self):
    # [FUNCTION DOC]
    # Scenario: a small generator emits assignments and returns as unparsed source and ifs as
    # brace-delimited blocks, run over a function with nested ifs after resolving origin info.
    # Asserts the generated buffer contains the expected statements and braces in order. The
    # source map is not checked (noted as a to-do in the code).

    class TestCodegen(transformer.CodeGenerator):
      # [CLASS DOC]
      # Minimal code generator with handlers for assignment, return and if nodes.

      def visit_Assign(self, node):
        # [FUNCTION DOC]
        # Emits the node's unparsed source followed by a newline.
        self.emit(parser.unparse(node, include_encoding_marker=False))
        self.emit('\n')

      def visit_Return(self, node):
        # [FUNCTION DOC]
        # Emits the node's unparsed source followed by a newline.
        self.emit(parser.unparse(node, include_encoding_marker=False))
        self.emit('\n')

      def visit_If(self, node):
        # [FUNCTION DOC]
        # Emits the condition as unparsed source in a brace-style header, then the visited body,
        # an else block with the visited else-branch, and closing braces.
        self.emit('if ')
        # This is just for simplifity. A real generator will walk the tree and
        # emit proper code.
        self.emit(parser.unparse(node.test, include_encoding_marker=False))
        self.emit(' {\n')
        self.visit_block(node.body)
        self.emit('} else {\n')
        self.visit_block(node.orelse)
        self.emit('}\n')

    tg = TestCodegen(self._simple_context())

    def test_fn():
      # [FUNCTION DOC]
      # HELPER: sample input function that is parsed, never called; has nested ifs.
      x = 1
      if x > 0:
        x = 2
        if x > 1:
          x = 3
      return x

    node, source = parser.parse_entity(test_fn, future_features=())
    origin_info.resolve(node, source, 'test_file', 100, 0)
    tg.visit(node)

    r = re.compile('.*'.join([
        r'x = 1',
        r'if \(?x > 0\)? {',
        r'x = 2',
        r'if \(?x > 1\)? {',
        r'x = 3',
        r'} else {',
        r'}',
        r'} else {',
        r'}',
        r'return x']), re.DOTALL)

    self.assertRegex(tg.code_buffer, r)
    # TODO(mdan): Test the source map.


if __name__ == '__main__':
  test.main()
