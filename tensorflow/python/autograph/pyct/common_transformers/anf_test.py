# [MODULE DOC]
# Unit tests for the A-normal-form conversion module. Provides a shared base test case with
# helpers (a minimal transformer context, AST comparison and a convert-and-compare routine that
# also checks idempotence), a main test case with many conversion scenarios, a subclass that
# re-runs them with a "transform nothing" configuration, and tests with custom configurations.
# Also has two top-level functions exercising conversion of the legacy exec statement.
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
"""Tests for anf module."""

import textwrap

import gast

from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.common_transformers import anf
from tensorflow.python.platform import test


# TODO(mdan): These two functions no longer need to be at the top level.
# TODO(mdan): Don't use exec.
def exec_test_function():
  # [FUNCTION DOC]
  # HELPER: test input for the exec conversion test. Contains an exec statement whose argument
  # is a nested concatenation (never actually run; it is only parsed).
  # The point is to test A-normal form conversion of exec
  # pylint: disable=exec-used
  exec('computed' + 5 + 'stuff', globals(), locals())


def exec_expected_result():
  # [FUNCTION DOC]
  # HELPER: expected output for the exec conversion test; each sub-expression of the exec
  # arguments is bound to a numbered temporary before the exec call.
  # pylint: disable=exec-used
  tmp_1001 = 'computed' + 5
  tmp_1002 = tmp_1001 + 'stuff'
  tmp_1003 = globals()
  tmp_1004 = locals()
  exec(tmp_1002, tmp_1003, tmp_1004)


class AnfTestBase(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base test case for the ANF tests; supplies context creation, AST comparison
  # and the convert-and-compare routine. Has no tests of its own.

  def _simple_context(self):
    # [FUNCTION DOC]
    # HELPER: builds a minimal transformer context (entity info with only a name and empty
    # future features, no namer, no user data). Returns it.
    entity_info = transformer.EntityInfo(
        name='test_fn',
        source_code=None,
        source_file=None,
        future_features=(),
        namespace=None)
    return transformer.Context(entity_info, None, None)

  def assert_same_ast(self, expected_node, node, msg=None):
    # [FUNCTION DOC]
    # HELPER: assertion. Unparses both ASTs to dedented, stripped source text and asserts the
    # texts are equal, with an optional failure message.
    expected_source = parser.unparse(expected_node, indentation='  ')
    expected_str = textwrap.dedent(expected_source).strip()
    got_source = parser.unparse(node, indentation='  ')
    got_str = textwrap.dedent(got_source).strip()
    self.assertEqual(expected_str, got_str, msg=msg)

  def assert_body_anfs_as_expected(self, expected_fn, test_fn, config=None):
    # [FUNCTION DOC]
    # HELPER: core assertion. Parses the input and expected functions, converts the input to ANF
    # (with the optional configuration), renames the result to match the expected function's name,
    # and asserts the two trees are the same. Then converts the result again and asserts nothing
    # changed (idempotence).
    # Testing the code bodies only.  Wrapping them in functions so the
    # syntax highlights nicely, but Python doesn't try to execute the
    # statements.
    exp_node, _ = parser.parse_entity(expected_fn, future_features=())
    node, _ = parser.parse_entity(test_fn, future_features=())
    node = anf.transform(node, self._simple_context(), config=config)
    exp_name = exp_node.name
    # Ignoring the function names in the result because they can't be
    # the same (because both functions have to exist in the same scope
    # at the same time).
    node.name = exp_name
    self.assert_same_ast(exp_node, node)
    # Check that ANF is idempotent
    node_repeated = anf.transform(node, self._simple_context())
    self.assert_same_ast(node_repeated, node)


class AnfTransformerTest(AnfTestBase):
  # [CLASS DOC]
  # Main test case for the ANF transformer with the default configuration. Each test gives an
  # input function and its expected converted form; the base helper does the comparison.

  def test_basic(self):
    # [FUNCTION DOC]
    # Converts a trivial function, loads the converted tree as a module and asserts the loaded
    # function returns the same value as the original.
    def test_function():
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      a = 0
      return a

    node, _ = parser.parse_entity(test_function, future_features=())
    node = anf.transform(node, self._simple_context())
    result, _, _ = loader.load_ast(node)
    self.assertEqual(test_function(), result.test_function())

  def test_binop_basic(self):
    # [FUNCTION DOC]
    # Chained additions are split so each intermediate sum gets its own temporary.

    def test_function(x, y, z):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      a = x + y + z
      return a

    def expected_result(x, y, z):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = x + y
      a = tmp_1001 + z
      return a

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_if_basic(self):
    # [FUNCTION DOC]
    # The condition's temporaries are hoisted before the if, while those of expressions in the
    # body stay inside the body.

    def test_function(a, b, c, e, f, g):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      if a + b + c:
        d = e + f + g
        return d

    def expected_result(a, b, c, e, f, g):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = a + b
      tmp_1002 = tmp_1001 + c
      if tmp_1002:
        tmp_1003 = e + f
        d = tmp_1003 + g
        return d

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_nested_binop_and_return(self):
    # [FUNCTION DOC]
    # A returned nested arithmetic expression is flattened into temporaries, evaluated
    # left to right, and the final temporary is returned.

    def test_function(b, c, d, e):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      return (2 * b + c) + (d + e)

    def expected_result(b, c, d, e):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = 2 * b
      tmp_1002 = tmp_1001 + c
      tmp_1003 = d + e
      tmp_1004 = tmp_1002 + tmp_1003
      return tmp_1004

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_function_call_and_expr(self):
    # [FUNCTION DOC]
    # A call statement with positional, keyword, starred and double-starred arguments gets each
    # argument expression bound to a temporary, in the order the transformer evaluates them.

    def test_function(call_something, a, b, y, z, c, d, e, f, g, h, i):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      call_something(a + b, y * z, kwarg=c + d, *(e + f), **(g + h + i))

    def expected_result(call_something, a, b, y, z, c, d, e, f, g, h, i):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = g + h
      tmp_1002 = a + b
      tmp_1003 = y * z
      tmp_1004 = e + f
      tmp_1005 = c + d
      tmp_1006 = tmp_1001 + i
      call_something(tmp_1002, tmp_1003, kwarg=tmp_1005, *tmp_1004, **tmp_1006)

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_with_and_print(self):
    # [FUNCTION DOC]
    # A with statement's context expression is hoisted before it, and the print argument inside
    # the body is flattened within the body.

    def test_function(a, b, c):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      with a + b + c as d:
        print(2 * d + 1)

    def expected_result(a, b, c):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = a + b
      tmp_1002 = tmp_1001 + c
      with tmp_1002 as d:
        tmp_1003 = 2 * d
        tmp_1004 = tmp_1003 + 1
        print(tmp_1004)

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_nested_multi_value_assign(self):
    # [FUNCTION DOC]
    # Tuple assignments and returns of nested tuples: non-trivial elements and nested tuples on
    # the right-hand side are bound to temporaries, while the target patterns stay intact.

    def test_function(a, b, c):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      x, y = a, a + b
      (z, y), x = (c, y + b), x + a
      return z, (y, x)

    def expected_result(a, b, c):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = a + b
      x, y = a, tmp_1001
      tmp_1002 = y + b
      tmp_1003 = (c, tmp_1002)
      tmp_1004 = x + a
      (z, y), x = tmp_1003, tmp_1004
      tmp_1005 = y, x
      tmp_1006 = z, tmp_1005
      return tmp_1006

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_deeply_nested_multi_value_assign(self):
    # [FUNCTION DOC]
    # A deeply nested list/tuple target is left untouched, while the same structure used as a
    # returned value is flattened bottom-up into temporaries.

    def test_function(a):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      [([(b, c), [d, e]], (f, g)), [(h, i, j), k]] = a
      return [([(b, c), [d, e]], (f, g)), [(h, i, j), k]]

    def expected_result(a):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      [([(b, c), [d, e]], (f, g)), [(h, i, j), k]] = a
      tmp_1001 = b, c
      tmp_1002 = [d, e]
      tmp_1003 = [tmp_1001, tmp_1002]
      tmp_1004 = f, g
      tmp_1005 = h, i, j
      tmp_1006 = tmp_1003, tmp_1004
      tmp_1007 = [tmp_1005, k]
      tmp_1008 = [tmp_1006, tmp_1007]
      return tmp_1008

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_local_definition_and_binary_compare(self):
    # [FUNCTION DOC]
    # A function containing a nested function definition: the inner body's arithmetic and
    # comparison are flattened into temporaries inside the inner function.

    def test_function():
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      def foo(a, b):
        # [FUNCTION DOC]
        # HELPER: inner function of the test input; compares twice the first argument to the second.
        return 2 * a < b
      return foo

    def expected_result():
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      def foo(a, b):
        # [FUNCTION DOC]
        # HELPER: inner function of the expected output, with temporaries for each sub-expression.
        tmp_1001 = 2 * a
        tmp_1002 = tmp_1001 < b
        return tmp_1002
      return foo

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_list_literal(self):
    # [FUNCTION DOC]
    # Each element expression of a returned list display is bound to a temporary, then the list
    # of temporaries is bound and returned.

    def test_function(a, b, c, d, e, f):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      return [a + b, c + d, e + f]

    def expected_result(a, b, c, d, e, f):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = a + b
      tmp_1002 = c + d
      tmp_1003 = e + f
      tmp_1004 = [tmp_1001, tmp_1002, tmp_1003]
      return tmp_1004

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_tuple_literal_and_unary(self):
    # [FUNCTION DOC]
    # A returned tuple containing a unary negation of a sum: inner expressions are flattened
    # and the tuple of temporaries is returned.

    def test_function(a, b, c, d, e, f):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      return (a + b, -(c + d), e + f)

    def expected_result(a, b, c, d, e, f):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = c + d
      tmp_1002 = a + b
      tmp_1003 = -tmp_1001
      tmp_1004 = e + f
      tmp_1005 = (tmp_1002, tmp_1003, tmp_1004)
      return tmp_1005

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_set_literal(self):
    # [FUNCTION DOC]
    # Arguments of a call to a set-building function are bound to temporaries before the call,
    # whose result is bound and returned.

    def test_function(a, b, c, d, e, f):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      return set(a + b, c + d, e + f)

    def expected_result(a, b, c, d, e, f):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = a + b
      tmp_1002 = c + d
      tmp_1003 = e + f
      tmp_1004 = set(tmp_1001, tmp_1002, tmp_1003)
      return tmp_1004

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_dict_literal_and_repr(self):
    # [FUNCTION DOC]
    # A dictionary display passed to a repr call: key and value expressions, the dictionary and
    # the call result are each bound to temporaries.

    def test_function(foo, bar, baz):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      return repr({foo + bar + baz: 7 | 8})

    def expected_result(foo, bar, baz):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = foo + bar
      tmp_1002 = tmp_1001 + baz
      tmp_1003 = 7 | 8
      tmp_1004 = {tmp_1002: tmp_1003}
      tmp_1005 = repr(tmp_1004)
      return tmp_1005

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_field_read_and_write(self):
    # [FUNCTION DOC]
    # Attribute chains: the base of the written target and each step of the read chain are bound
    # to temporaries; the final assignment target attribute stays in place.

    def test_function(a, d):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      a.b.c = d.e.f + 3

    def expected_result(a, d):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = a.b
      tmp_1002 = d.e
      tmp_1003 = tmp_1002.f
      tmp_1001.c = tmp_1003 + 3

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_subscript_read_and_write(self):
    # [FUNCTION DOC]
    # Same as the attribute test but for subscripts: intermediate subscripts are bound to
    # temporaries, the final written subscript target stays in place.

    def test_function(a, b, c, d, e, f):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      a[b][c] = d[e][f] + 3

    def expected_result(a, b, c, d, e, f):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = a[b]
      tmp_1002 = d[e]
      tmp_1003 = tmp_1002[f]
      tmp_1001[c] = tmp_1003 + 3

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_augassign_and_delete(self):
    # [FUNCTION DOC]
    # Augmented assignment keeps its target and right-hand side form (inner sub-expressions are
    # still named); deleting a plain name is unchanged, and deleting a nested subscript names
    # only the intermediate subscript.

    def test_function(a, x, y, z):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      a += x + y + z
      del a
      del z[y][x]

    def expected_result(a, x, y, z):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = x + y
      a += tmp_1001 + z
      del a
      tmp_1002 = z[y]
      del tmp_1002[x]

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_raise_yield_and_raise(self):
    # [FUNCTION DOC]
    # A yielded power expression and a raised call (whose argument is a concatenation) are
    # flattened into temporaries before the yield and raise statements.
    # KNOWN ISSUE: the name says "raise_yield_and_raise" (raise listed twice); it covers one
    # yield and one raise.

    def test_function(a, c, some_computed, exception):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      yield a ** c
      raise some_computed('complicated' + exception)

    def expected_result(a, c, some_computed, exception):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = a ** c
      yield tmp_1001
      tmp_1002 = 'complicated' + exception
      tmp_1003 = some_computed(tmp_1002)
      raise tmp_1003

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_with_and_if_with_expressions(self):
    # [FUNCTION DOC]
    # A with statement and an if statement whose headers are expressions: each header expression
    # is hoisted before its statement, and body calls get their argument temporaries inside.

    def test_function(foo, bar, function, quux, quozzle, w, x, y, z):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      with foo + bar:
        function(x + y)
      if quux + quozzle:
        function(z / w)

    def expected_result(foo, bar, function, quux, quozzle, w, x, y, z):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = foo + bar
      with tmp_1001:
        tmp_1002 = x + y
        function(tmp_1002)
      tmp_1003 = quux + quozzle
      if tmp_1003:
        tmp_1004 = z / w
        function(tmp_1004)

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_exec(self):
    # [FUNCTION DOC]
    # Compares the conversion of the module-level exec example function with its module-level
    # expected counterpart.
    self.assert_body_anfs_as_expected(exec_expected_result, exec_test_function)

  def test_simple_while_and_assert(self):
    # [FUNCTION DOC]
    # A while loop with a trivial test and an assert of a plain name: only the multiplication in
    # the body is hoisted (inside the body); the loop test and assert stay as they are.

    def test_function(foo, quux):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      while foo:
        assert quux
        foo = foo + 1 * 3

    def expected_result(foo, quux):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      while foo:
        assert quux
        tmp_1001 = 1 * 3
        foo = foo + tmp_1001

    self.assert_body_anfs_as_expected(expected_result, test_function)

  def test_for(self):
    # [FUNCTION DOC]
    # A for loop whose iterable is a call with a sum argument: the iterable's temporaries are
    # hoisted before the loop; body expressions are flattened inside the body.

    def test_function(compute, something, complicated, foo):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      for foo in compute(something + complicated):
        bar = foo + 1 * 3
      return bar

    def expected_result(compute, something, complicated, foo):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = something + complicated
      tmp_1002 = compute(tmp_1001)
      for foo in tmp_1002:
        tmp_1003 = 1 * 3
        bar = foo + tmp_1003
      return bar

    self.assert_body_anfs_as_expected(expected_result, test_function)

  # This test collects several examples where the definition of A-normal form
  # implemented by this transformer is questionable.  Mostly it's here to spell
  # out what the definition is in these cases.
  def test_controversial(self):
    # [FUNCTION DOC]
    # Documents questionable cases of the ANF definition: assignments to attributes, subscripts
    # and tuples and augmented assignment keep their right-hand sides as-is; calls and subscript
    # indices get temporaries. Compares against the stated expected form.

    def test_function(b, c, d, f):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      a = c + d
      a.b = c + d
      a[b] = c + d
      a += c + d
      a, b = c
      a, b = c, d
      a = f(c)
      a = f(c + d)
      a[b + d] = f.e(c + d)

    def expected_result(b, c, d, f):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      a = c + d
      a.b = c + d  # Should be a.b = tmp?  (Definitely not tmp = c + d)
      a[b] = c + d  # Should be a[b] = tmp?  (Definitely not tmp = c + d)
      a += c + d  # Should be a += tmp?  (Definitely not tmp = c + d)
      a, b = c  # Should be a = c[0], b = c[1]?  Or not?
      a, b = c, d  # Should be a = c, b = d?  Or not?
      a = f(c)
      tmp_1001 = c + d
      a = f(tmp_1001)
      tmp_1002 = b + d
      tmp_1003 = f.e
      tmp_1004 = c + d
      a[tmp_1002] = tmp_1003(tmp_1004)  # Or should be a[tmp1] = tmp2?

    self.assert_body_anfs_as_expected(expected_result, test_function)


class AnfNonTransformationTest(AnfTransformerTest):
  # [CLASS DOC]
  # Re-runs every scenario of the main ANF test case with a configuration that leaves every
  # edge alone, by overriding the comparison helper to assert the source is unchanged.
  """Test that specifying "no transformation" does nothing.

  Reuses all the examples of AnfTransformerTest by overriding
  `assert_body_anfs_as_expected_`.
  """

  def assert_body_anfs_as_expected(self, expected_fn, test_fn):
    # [FUNCTION DOC]
    # Override of the comparison helper. Ignores the expected function; transforms the input
    # with a configuration that leaves everything alone and asserts the printed source is
    # identical before and after.
    # KNOWN ISSUE: the override drops the optional configuration parameter of the base helper, and
    # the docstring refers to a name with a trailing underscore that does not exist.
    # Testing the code bodies only.  Wrapping them in functions so the
    # syntax highlights nicely, but Python doesn't try to execute the
    # statements.
    node, _ = parser.parse_entity(test_fn, future_features=())
    orig_source = parser.unparse(node, indentation='  ')
    orig_str = textwrap.dedent(orig_source).strip()
    config = [(anf.ANY, anf.LEAVE)]  # Configuration to transform nothing
    node = anf.transform(node, self._simple_context(), config=config)
    new_source = parser.unparse(node, indentation='  ')
    new_str = textwrap.dedent(new_source).strip()
    self.assertEqual(orig_str, new_str)


class AnfConfiguredTest(AnfTestBase):
  # [CLASS DOC]
  # Tests ANF conversion with custom configurations that differ from the default.

  def test_constants_in_function_calls(self):
    # [FUNCTION DOC]
    # Configuration that replaces only literal and name arguments of calls: constants passed to
    # a call are moved to temporaries while compound expressions stay in place.
    # An example specific configuration that differs from the default: Moving
    # literals out of being directly passed to functions, but nothing else.
    try:
      # TODO(b/140808434): Fix this.
      # gast pre-0.3
      literals = (gast.Num, gast.Str, gast.Bytes, gast.NameConstant, gast.Name)
    except AttributeError:
      # gast 0.3+
      literals = (gast.Constant, gast.Name)
    config = [(anf.ASTEdgePattern(gast.Call, anf.ANY, literals), anf.REPLACE)]

    def test_function(x, frob):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      return frob(x, x+1, 2)

    def expected_result(x, frob):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = 2
      return frob(x, x+1, tmp_1001)

    self.assert_body_anfs_as_expected(expected_result, test_function, config)

  def test_anf_some_function_calls(self):
    # [FUNCTION DOC]
    # Configuration using a callable decision: arguments are moved out only for calls to
    # functions in an allow list (here one name), other calls are left unchanged.
    # Another example specific configuration that differs from the default:
    # Moving all arguments out of some function calls but leaving others be.
    allowlist = ['foo']

    def transform(parent, field, child):
      # [FUNCTION DOC]
      # HELPER: configuration callable for the test. Ignores field and child; returns true if the
      # called function's name (read from the parent call node) is in the allow list.
      del field
      del child
      func_name = parent.func.id
      return str(func_name) in allowlist

    config = [(anf.ASTEdgePattern(gast.Call, anf.ANY, anf.ANY), transform)]

    def test_function(x, foo, bar):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      y = foo(x, x+1, 2)
      return bar(y, y+1, 2)

    def expected_result(x, foo, bar):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = x+1
      tmp_1002 = 2
      y = foo(x, tmp_1001, tmp_1002)
      return bar(y, y+1, 2)

    self.assert_body_anfs_as_expected(expected_result, test_function, config)

  def test_touching_name_constant(self):
    # [FUNCTION DOC]
    # Configuration that replaces name and constant arguments of calls: true, false and none
    # arguments are moved to temporaries, showing configurations can affect such nodes.
    # Checking that the nodes for `True`, `False`, and `None` can be manipulated
    # by a configuration.  This is non-trivial, because in Python 2 those are
    # represented as `Name`, which is the same node type as variable references.
    specials = (gast.Name, gast.Constant)
    config = [(anf.ASTEdgePattern(gast.Call, anf.ANY, specials), anf.REPLACE)]

    def test_function(f):
      # [FUNCTION DOC]
      # HELPER: test input; the function whose source is converted to A-normal form by the test.
      return f(True, False, None)

    def expected_result(f):
      # [FUNCTION DOC]
      # HELPER: test expectation; the function whose source is the expected A-normal-form output.
      tmp_1001 = True
      tmp_1002 = False
      tmp_1003 = None
      return f(tmp_1001, tmp_1002, tmp_1003)

    self.assert_body_anfs_as_expected(expected_result, test_function, config)


if __name__ == '__main__':
  test.main()
