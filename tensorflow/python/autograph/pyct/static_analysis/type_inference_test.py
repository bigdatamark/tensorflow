# [MODULE DOC]
# Unit tests for the autograph static type-inference pass (type_inference module).
# Each test defines a small Resolver subclass (the user-supplied oracle that maps names, values,
# arguments, calls, slices and operators to sets of types), a tiny function to analyze, runs it
# through a test transpiler that applies the full analysis pipeline (qualified names, activity,
# control-flow graph, reaching definitions, reaching function definitions, type inference), and
# then asserts on the type and value annotations attached to the resulting syntax tree nodes.
# No import-time side effects other than the test main entry point when run as a script.
# NOTE: the small functions that are parsed and analyzed by the transpiler (named test_fn, g,
# foo, bar and local_fn) are intentionally left without documentation comments because their
# source text is re-parsed by the code under test.
#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for type_inference module."""

from typing import Any, Callable, List

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
from tensorflow.python.autograph.pyct.static_analysis import type_inference
from tensorflow.python.platform import test


class BasicTestResolver(type_inference.Resolver):
  # [CLASS DOC]
  # HELPER: shared resolver used by several tests in this module.
  # Minimal type resolver: names are looked up in the supplied namespace and typed by their
  # runtime type (the name for integers is special-cased to the integer type itself), values are
  # typed by their runtime type, and arguments are typed by the string form of their annotation.
  """A very basic resolver for testing."""

  def res_name(self, ns, types_ns, name):
    # [FUNCTION DOC]
    # Resolves a name. Inputs: namespace mapping, types namespace (unused), qualified name.
    # Returns a pair: a set holding the runtime type of the namespace entry, and the entry itself.
    # The name for the integer builtin yields the set of the integer type and the integer type.
    # Raises a key error if the name is absent from the namespace.
    str_name = str(name)
    if str_name == 'int':
      return {int}, int
    return {type(ns[str_name])}, ns[str_name]

  def res_value(self, ns, value):
    # [FUNCTION DOC]
    # Resolves a static value: returns a set holding the runtime type of the value.
    return {type(value)}

  def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
    # [FUNCTION DOC]
    # Resolves a function argument. Returns nothing (null) if the argument has no type
    # annotation, otherwise a set containing the string form of the annotation.
    if type_anno is None:
      return None
    return {str(type_anno)}


class TestTranspiler(transpiler.GenericTranspiler):
  # [CLASS DOC]
  # HELPER: test transpiler that runs the analysis pipeline for the type inference tests.
  # Generic transpiler that, instead of generating code, applies the analyses (qualified names,
  # activity, control flow graph, reaching definitions, reaching function definitions and type
  # inference) and returns the annotated tree. Constructed with a resolver class, which it
  # instantiates and keeps as its resolver attribute.

  def __init__(self, resolver_type):
    # [FUNCTION DOC]
    # Creates the transpiler; resolver_type is a resolver class that is instantiated with no
    # arguments and stored for use during tree transformation.
    super().__init__()
    self.resolver = resolver_type()

  def get_transformed_name(self, _):
    # [FUNCTION DOC]
    # Returns the fixed name used for the transformed entity regardless of the input node.
    return 'test_item'

  def transform_ast(self, node, ctx):
    # [FUNCTION DOC]
    # Runs the analyses in order on the function tree node using the given context, and returns
    # the tree annotated with types (and values) by the type inference pass.
    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx)
    graphs = cfg.build(node)
    node = reaching_definitions.resolve(node, ctx, graphs)
    node = reaching_fndefs.resolve(node, ctx, graphs)
    node = type_inference.resolve(node, ctx, graphs, self.resolver)
    return node


class TypeInferenceAnalyzerTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the type inference analyzer: argument typing, assignment propagation,
  # attribute and property resolution, closures of local functions, calls with side effects,
  # subscripts, unpacking, operators and literals. Provides two assertion helpers.

  def assertTypes(self, node, expected):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the tests in this class.
    # Asserts that the set of types annotated on the given tree node equals the expected types.
    # expected may be a single type or a tuple of types (a single value is wrapped in a tuple).
    if not isinstance(expected, tuple):
      expected = expected,
    self.assertSetEqual(
        set(anno.getanno(node, anno.Static.TYPES)), set(expected))

  def assertClosureTypes(self, node, expected):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the tests in this class.
    # Reads the closure types annotation of the node (mapping from symbols to type sets),
    # converts keys to strings, and asserts that every expected key is present with exactly
    # the expected value. Extra keys in the annotation are allowed.
    actual = anno.getanno(node, anno.Static.CLOSURE_TYPES)
    actual = {str(k): v for k, v in actual.items()}
    for k, v in expected.items():
      self.assertIn(k, actual)
      self.assertEqual(actual[k], v)

  def test_no_inference_on_unknown_operand_types(self):
    # [FUNCTION DOC]
    # Scenario: resolver returns no type for arguments, and the function compares and subtracts
    # them. Expected: no type annotation is attached to either resulting operator expression.

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver that reports no type for any argument.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Always returns nothing (null), meaning the argument type is unknown.
        return None

    def test_fn(a, b):
      return a < b, a - b

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    # With no information on operand types, the operators will infer nothing.
    self.assertFalse(
        anno.hasanno(fn_body[0].value.elts[0], anno.Static.TYPES))
    self.assertFalse(
        anno.hasanno(fn_body[0].value.elts[1], anno.Static.TYPES))

  def test_resolver_output_checked(self):
    # [FUNCTION DOC]
    # Scenario: a resolver returns a non-set value (an integer) for an argument type.
    # Expected: the transformation raises a value error whose message says a set was expected.

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver returning an invalid (non-set) argument type to trigger output validation.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns an integer instead of a set of types, which is invalid.
        return 1

    def test_fn(a):
      del a
      pass

    with self.assertRaisesRegex(ValueError, 'expected to return set'):
      TestTranspiler(Resolver).transform(test_fn, None)

  def test_argument(self):
    # [FUNCTION DOC]
    # Scenario: the resolver is called for the arguments of an annotated top-level function.
    # Expected: the argument resolution is told the function is not local, receives the
    # annotation for the first argument, and the returned types ("<name>_type") are attached
    # to the uses of the arguments in the return statement.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver that checks the arguments passed to it and returns "<name>_type" as the type.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Asserts the function is not local and, for the argument named a, that its annotation is
        # the integer name. Returns a set with one string derived from the argument name.
        test_self.assertFalse(f_is_local)
        if name == qual_names.QN('a'):
          test_self.assertEqual(type_anno, qual_names.QN('int'))
        return {str(name) + '_type'}

    def test_fn(a: int, b):
      return a, b

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.elts[0], 'a_type')
    self.assertTypes(fn_body[0].value.elts[1], 'b_type')

  def test_argument_of_local_function(self):
    # [FUNCTION DOC]
    # Scenario: an annotated function defines a nested local function with its own arguments.
    # Expected: the resolver sees the outer function as non-local and the nested one as local,
    # with the right annotations, and types flow through the call to the nested function.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver that validates the function names, locality flags and annotations of arguments.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Checks arguments per function: for the outer function expects non-local and an integer
        # annotation on a; for the nested function expects local, a float annotation on x and no
        # annotation on y. Fails the test for anything unexpected. Returns a one-string type set
        # derived from the argument name.
        if f_name == 'test_fn':
          test_self.assertFalse(f_is_local)
          test_self.assertEqual(name, qual_names.QN('a'))
          test_self.assertEqual(type_anno, qual_names.QN('int'))
        elif f_name == 'foo':
          test_self.assertTrue(f_is_local)
          if name == qual_names.QN('x'):
            test_self.assertEqual(type_anno, qual_names.QN('float'))
          elif name == qual_names.QN('y'):
            test_self.assertIsNone(type_anno)
          else:
            test_self.fail('unexpected argument {} for {}'.format(name, f_name))
        else:
          test_self.fail('unexpected function name {}'.format(f_name))
        return {str(name) + '_type'}

    def test_fn(a: int):

      def foo(x: float, y):
        return x, y

      return foo(a, a)

    tr = TestTranspiler(Resolver)
    node, _ = tr.transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[0].value, (('x_type', 'y_type'),))
    self.assertTypes(fn_body[0].body[0].value.elts[0], 'x_type')
    self.assertTypes(fn_body[0].body[0].value.elts[1], 'y_type')

  def test_assign_straightline(self):
    # [FUNCTION DOC]
    # Scenario: copy an integer argument into a variable and return it with a float argument.
    # Expected: types propagate through the assignment (integer for the copy, float for the
    # float argument) on both the targets and the uses.

    def test_fn(a: int, c: float):
      b = a
      return a, b, c

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].targets[0], 'int')
    self.assertTypes(fn_body[0].value, 'int')
    self.assertTypes(fn_body[1].value.elts[0], 'int')
    self.assertTypes(fn_body[1].value.elts[1], 'int')
    self.assertTypes(fn_body[1].value.elts[2], 'float')

  def test_expr(self):
    # [FUNCTION DOC]
    # Scenario: an expression statement calls a method of a global object.
    # Expected: the attribute is typed by value resolution (string type), the call by call
    # resolution (integer type), the static value of the function node is the bound method, and
    # the expression statement itself receives the call's type.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver for a global object: types the name, its attribute value and the call result.

      def res_value(self, ns, value):
        # [FUNCTION DOC]
        # Asserts the value is the method of the global object and returns the string type.
        test_self.assertEqual(value, tc.a)
        return {str}

      def res_name(self, ns, types_ns, name):
        # [FUNCTION DOC]
        # Asserts the name is the global object name; returns its class type and the object.
        test_self.assertEqual(name, qual_names.QN('tc'))
        return {TestClass}, tc

      def res_call(self, ns, types_ns, node, f_type, args, keywords):
        # [FUNCTION DOC]
        # Asserts the callee type is the string type; returns the integer type and no side effects.
        test_self.assertEqual(f_type, (str,))
        return {int}, None

    class TestClass:
      # [CLASS DOC]
      # Simple class with one method, used as the type of the global object.

      def a(self):
        # [FUNCTION DOC]
        # Empty method used only as a call target.
        pass

    tc = TestClass()

    def test_fn():
      tc.a()

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertEqual(
        anno.getanno(fn_body[0].value.func, anno.Static.VALUE), tc.a)
    self.assertTypes(fn_body[0].value.func, str)
    self.assertTypes(fn_body[0].value, int)
    self.assertTypes(fn_body[0], int)

  def test_assign_overwriting(self):
    # [FUNCTION DOC]
    # Scenario: a variable is assigned an integer argument and then a float argument.
    # Expected: each assignment target and value carries the type of its own right-hand side,
    # i.e. the later assignment overwrites the earlier type.

    def test_fn(a: int, b: float):
      c = a
      c = b
      return c

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].targets[0], 'int')
    self.assertTypes(fn_body[0].value, 'int')
    self.assertTypes(fn_body[1].targets[0], 'float')
    self.assertTypes(fn_body[1].value, 'float')

  def test_dynamic_attribute_of_static_value(self):
    # [FUNCTION DOC]
    # Scenario: reading an instance attribute (set in the constructor) of a global object whose
    # value is known. Expected: the object keeps its class type and static value, the attribute
    # gets the type from value resolution (integer) and its actual value as static value.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver that knows the global object's value, so attributes are resolved by value.

      def res_value(self, ns, value):
        # [FUNCTION DOC]
        # Asserts the value is the attribute of the global object; returns the integer type.
        test_self.assertEqual(value, tc.a)
        return {int}

      def res_name(self, ns, types_ns, name):
        # [FUNCTION DOC]
        # Asserts the name is the global object; returns its class type and the object.
        test_self.assertEqual(name, qual_names.QN('tc'))
        return {TestClass}, tc

    class TestClass:
      # [CLASS DOC]
      # Class with an instance attribute assigned in its constructor.

      def __init__(self):
        # [FUNCTION DOC]
        # Sets the instance attribute a to the number one.
        self.a = 1

    tc = TestClass()

    def test_fn():
      return tc.a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.value, TestClass)
    self.assertTypes(fn_body[0].value, int)
    self.assertIs(anno.getanno(fn_body[0].value.value, anno.Static.VALUE), tc)
    self.assertEqual(anno.getanno(fn_body[0].value, anno.Static.VALUE), tc.a)

  def test_static_attribute_of_typed_value(self):
    # [FUNCTION DOC]
    # Scenario: reading a class-level attribute of an object known only by type (no value).
    # Expected: the object has the class type but no static value; the attribute is typed by the
    # resolver (string, which takes precedence) and carries the class attribute as static value.

    test_self = self

    class TestClass:
      # [CLASS DOC]
      # Class with a class-level attribute a equal to one.

      a = 1

    tc = TestClass()

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver returning only a type (no value) for the object name.

      def res_name(self, ns, types_ns, name):
        # [FUNCTION DOC]
        # Asserts the name is the object; returns its class type and no value.
        test_self.assertEqual(name, qual_names.QN('tc'))
        return {TestClass}, None

      def res_value(self, ns, value):
        # [FUNCTION DOC]
        # Asserts the value is the class attribute; returns the string type.
        test_self.assertIs(value, tc.a)
        return {str}

    def test_fn():
      return tc.a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.value, TestClass)
    self.assertTypes(fn_body[0].value, str)  # Resolver is SOT
    self.assertFalse(anno.hasanno(fn_body[0].value.value, anno.Static.VALUE))
    self.assertEqual(anno.getanno(fn_body[0].value, anno.Static.VALUE), 1)

  def test_static_attribute_of_ambiguous_type(self):
    # [FUNCTION DOC]
    # Scenario: the object has two candidate class types with different attribute values.
    # Expected: the object keeps both types, but the attribute gets neither a type nor a value,
    # since the lookup is ambiguous.

    test_self = self

    class TestClass1:
      # [CLASS DOC]
      # First candidate class; attribute a equals one.

      a = 1

    class TestClass2:
      # [CLASS DOC]
      # Second candidate class; attribute a equals two.

      a = 2

    tc = TestClass1()

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver reporting two possible classes for the object name.

      def res_name(self, ns, types_ns, name):
        # [FUNCTION DOC]
        # Asserts the name is the object; returns both class types and no value.
        test_self.assertEqual(name, qual_names.QN('tc'))
        return {TestClass1, TestClass2}, None

      def res_value(self, ns, value):
        # [FUNCTION DOC]
        # Accepts values one or two; returns the string type.
        test_self.assertIn(value, (1, 2))
        return {str}

    def test_fn():
      return tc.a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.value, (TestClass1, TestClass2))
    self.assertFalse(anno.hasanno(fn_body[0].value, anno.Static.TYPES))
    self.assertFalse(anno.hasanno(fn_body[0].value.value, anno.Static.VALUE))
    self.assertFalse(anno.hasanno(fn_body[0].value, anno.Static.VALUE))

  def test_property_of_typed_value(self):
    # [FUNCTION DOC]
    # Scenario: reading a property of an object known only by type.
    # Expected: the resolver receives the property object of the class (not the evaluated
    # result), the attribute gets the property type and the property object as static value.

    test_self = self

    class TestClass:
      # [CLASS DOC]
      # Class with a read-only property a that returns one.

      @property
      def a(self):
        # [FUNCTION DOC]
        # Property getter returning the number one.
        return 1

    tc = TestClass()

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver returning only a type for the object, and the property type for the attribute.

      def res_name(self, ns, types_ns, name):
        # [FUNCTION DOC]
        # Asserts the name is the object; returns its class type and no value.
        test_self.assertEqual(name, qual_names.QN('tc'))
        return {TestClass}, None

      def res_value(self, ns, value):
        # [FUNCTION DOC]
        # Asserts the value is the class's property object (not the number one); returns the
        # property type.
        test_self.assertIs(value, TestClass.a)
        test_self.assertNotEqual(value, 1)  # Can't evaluate property of class.
        return {property}

    def test_fn():
      return tc.a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.value, TestClass)
    self.assertTypes(fn_body[0].value, property)
    self.assertFalse(anno.hasanno(fn_body[0].value.value, anno.Static.VALUE))
    self.assertEqual(
        anno.getanno(fn_body[0].value, anno.Static.VALUE), TestClass.a)

  def test_dynamic_attribute_of_typed_value(self):
    # [FUNCTION DOC]
    # Scenario: reading an instance attribute (only set at construction) of an object known only
    # by type. Expected: the object has its class type; the attribute has no type and no value,
    # since it cannot be resolved statically.

    test_self = self

    class TestClass:
      # [CLASS DOC]
      # Class whose attribute a exists only on instances.

      def __init__(self):
        # [FUNCTION DOC]
        # Sets the instance attribute a to the number one.
        self.a = 1

    tc = TestClass()

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver returning only a class type for the object name.

      def res_name(self, ns, types_ns, name):
        # [FUNCTION DOC]
        # Asserts the name is the object; returns its class type and no value.
        test_self.assertEqual(name, qual_names.QN('tc'))
        return {TestClass}, None

    def test_fn():
      return tc.a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.value, TestClass)
    self.assertFalse(anno.hasanno(fn_body[0].value, anno.Static.TYPES))
    self.assertFalse(anno.hasanno(fn_body[0].value.value, anno.Static.VALUE))
    self.assertFalse(anno.hasanno(fn_body[0].value, anno.Static.VALUE))

  def test_external_value(self):
    # [FUNCTION DOC]
    # Scenario: a function reads a closure string variable and returns a copy.
    # Expected: with the basic resolver the assignment target and returned value are typed string.

    a = 'foo'

    def test_fn():
      b = a
      return b

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].targets[0], str)
    self.assertTypes(fn_body[1].value, str)

  def test_external_function(self):
    # [FUNCTION DOC]
    # Scenario: calling an external function annotated to return float.
    # Expected: the function name node is typed by name resolution (string), and the call result,
    # assignment target and returned value are typed by call resolution (float).

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver that types an external function name and its call.

      def res_name(self, ns, types_ns, name):
        # [FUNCTION DOC]
        # Asserts the name is g; returns the string type and the function itself.
        test_self.assertEqual(name, qual_names.QN('g'))
        return {str}, g

      def res_call(self, ns, types_ns, node, f_type, args, keywords):
        # [FUNCTION DOC]
        # Asserts the callee type is the string type and the call node's function carries the
        # qualified name g; returns the float type and no side effects.
        test_self.assertEqual(f_type, (str,))
        test_self.assertEqual(
            anno.getanno(node.func, anno.Basic.QN), qual_names.QN('g'))
        return {float}, None

    def g() -> float:
      return 1.0

    def test_fn():
      a = g()
      return a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value.func, str)
    self.assertTypes(fn_body[0].targets[0], float)
    self.assertTypes(fn_body[1].value, float)

  def test_external_function_side_effects(self):
    # [FUNCTION DOC]
    # Scenario: a call to an external function whose resolution reports a side effect, namely
    # that the caller's variable x becomes a string. Expected: before the call x has the
    # annotated integer type; after the call, x is a string while the earlier copy y stays integer.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver that returns no type for the callee but reports a side effect on variable x.

      def res_name(self, ns, types_ns, name):
        # [FUNCTION DOC]
        # Asserts the name is g; returns no types and the function itself.
        test_self.assertEqual(name, qual_names.QN('g'))
        return None, g

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns a set with the string form of the argument annotation.
        return {str(type_anno)}

      def res_call(self, ns, types_ns, node, f_type, args, keywords):
        # [FUNCTION DOC]
        # Asserts the callee has no known type; returns no result type and a mapping saying that
        # symbol x now has the string type.
        test_self.assertIsNone(f_type)
        return None, {qual_names.QN('x'): {str}}

    def g():
      # The resolver will pretend that this function has the following body:
      #
      #   nonlocal x
      #   x = 'a'
      pass

    def test_fn(x: int):
      y = x
      g()
      return x, y

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].targets[0], 'int')
    self.assertTypes(fn_body[0].value, 'int')
    self.assertTypes(fn_body[2].value.elts[0], str)
    self.assertTypes(fn_body[2].value.elts[1], 'int')

  def test_local_function_closure(self):
    # [FUNCTION DOC]
    # Scenario: a nested function reads a variable of its enclosing function.
    # Expected: the use inside the nested function is typed integer, and the nested function's
    # closure types annotation maps x to the integer type.

    def test_fn(x: int):

      def foo():
        return x

      foo()

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[0].value, 'int')
    self.assertClosureTypes(fn_body[0], {'x': {'int'}})

  def test_local_function_closure_nested(self):
    # [FUNCTION DOC]
    # Scenario: two levels of nested functions where the innermost reads the outermost argument.
    # Expected: the use is typed integer and both enclosing local functions list x as a closure
    # variable of integer type.

    def test_fn(x: int):

      def foo():

        def bar():
          return x

        bar()

      foo()

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[0].body[0].value, 'int')
    self.assertClosureTypes(fn_body[0], {'x': {'int'}})
    self.assertClosureTypes(fn_body[0].body[0], {'x': {'int'}})

  def test_local_function_closure_mutable_var(self):
    # [FUNCTION DOC]
    # Scenario: a nested function declares the enclosing variable nonlocal and reads it.
    # Expected: the use is typed integer and the closure types record x as integer.

    def test_fn(x: int):

      def foo():
        nonlocal x
        return x

      foo()

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[1].value, 'int')
    self.assertClosureTypes(fn_body[0], {'x': {'int'}})

  def test_local_function_closure_ignored_for_bound_symbols(self):
    # [FUNCTION DOC]
    # Scenario: a nested function assigns to a name x that is also an outer argument, so x is
    # local to the nested function. Expected: the read of x on the right-hand side gets no type,
    # but the closure types still list x as float (taken from the outer scope).

    def test_fn(x: float):  # pylint:disable=unused-argument

      def foo():
        x = x + 1  # pylint:disable=used-before-assignment

      foo()

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertFalse(
        anno.hasanno(fn_body[0].body[0].value.left, anno.Static.TYPES))
    self.assertClosureTypes(fn_body[0], {'x': {'float'}})

  def test_local_function_closure_uses_call_site_types(self):
    # [FUNCTION DOC]
    # Scenario: the outer variable is rebound to a float after the nested function is defined and
    # before it is called. Expected: types inside the nested function and in its closure use the
    # types at the call site (float), not at the definition site (integer).

    def test_fn(x: int):

      def foo():
        return x

      x = 1.0
      foo()

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[0].value, float)
    self.assertTypes(fn_body[1].targets[0], float)
    self.assertClosureTypes(fn_body[0], {'x': {float}})

  def test_local_function_hides_locals(self):
    # [FUNCTION DOC]
    # Scenario: a nested function assigns a local variable with the same name as an outer one.
    # Expected: the nested function's local assignment target receives no type annotation
    # (outer types are not leaked into the nested function's own locals).

    def test_fn(a: int):  # pylint:disable=unused-argument

      def local_fn(v):
        a = v
        return a

      local_fn(1)

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertFalse(
        anno.hasanno(fn_body[0].body[0].targets[0], anno.Static.TYPES))

  def test_local_function_type(self):
    # [FUNCTION DOC]
    # Scenario: a nested function annotated to return integer is defined and called.
    # Expected: the function name at the call site is typed as a callable taking anything and
    # returning integer; the call and the expression statement are typed integer.

    def test_fn(x: int):

      def foo() -> int:
        return x

      foo()

    node, _ = TestTranspiler(BasicTestResolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[1].value.func, Callable[[Any], int])
    self.assertTypes(fn_body[1].value, int)
    self.assertTypes(fn_body[1], int)

  def test_side_effects_on_arg_function_closure(self):
    # [FUNCTION DOC]
    # Scenario: a local function that reads a variable is passed to an external function whose
    # resolved side effect is rebinding that variable to a string. Expected: the variable read
    # inside the local function is typed string, i.e. side effects of calls taking the local
    # function as an argument update the closure types.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver for an external function taking a callable and mutating variable x.

      def res_name(self, ns, types_ns, name):
        # [FUNCTION DOC]
        # Asserts the name is g; returns a callable type taking a callable and returning nothing,
        # plus the function itself.
        test_self.assertEqual(name, qual_names.QN('g'))
        return {Callable[[Callable], None]}, g

      def res_value(self, ns, value):
        # [FUNCTION DOC]
        # Asserts the value is the float one; returns the float type.
        test_self.assertEqual(value, 1.0)
        return {float}

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns a set with the string form of the argument annotation.
        return {str(type_anno)}

      def res_call(self, ns, types_ns, node, f_type, args, keywords):
        # [FUNCTION DOC]
        # Asserts the call is to g with the expected callable type; returns no result type and a
        # side-effect mapping setting x to the string type.
        test_self.assertEqual(node.func.id, 'g')
        test_self.assertEqual(f_type, (Callable[[Callable], None],))
        return None, {qual_names.QN('x'): {str}}

    def g(foo):
      # The resolver will convey that this function has the following body:
      #
      #   nonlocal x
      #   x = 'a'
      #   foo()
      del foo
      pass

    def test_fn(x: int):  # pylint:disable=unused-argument

      def foo():
        return x

      x = 1.0
      g(foo)

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].body[0].value, str)

  def test_subscript(self):
    # [FUNCTION DOC]
    # Scenario: an argument typed list is indexed by a literal. Expected: slice resolution is
    # called with the list type and the integer index type, and its string result becomes the
    # type of the subscript expression, with component types on the value and slice nodes.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver typing arguments as list, values as integer, and subscripts as string.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns the list type for every argument.
        return {list}

      def res_value(self, ns, value):
        # [FUNCTION DOC]
        # Returns the integer type for every value.
        return {int}

      def res_slice(self, ns, types_ns, node, value, slice_):
        # [FUNCTION DOC]
        # Asserts the indexed type set is list and the index type set is integer; returns string.
        test_self.assertSetEqual(value, {list})
        test_self.assertSetEqual(slice_, {int})
        return {str}

    def test_fn(a):
      return a[1]

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, str)
    self.assertTypes(fn_body[0].value.value, list)
    self.assertTypes(fn_body[0].value.slice, int)

  def test_tuple_unpacking(self):
    # [FUNCTION DOC]
    # Scenario: a list-typed argument is unpacked into two variables. Expected: the unpacking is
    # resolved as indexing at positions zero and one (slice resolution receives the position),
    # giving float and string, and the returned tuple is typed as a tuple of those.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver that returns float for position zero and string for position one on slicing.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns the list type for every argument.
        return {list}

      def res_value(self, ns, value):
        # [FUNCTION DOC]
        # Returns the integer type for every value.
        return {int}

      def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
        # [FUNCTION DOC]
        # Asserts the position is zero or one, the indexed type is list and the index type is
        # integer; returns float for position zero and string for position one.
        test_self.assertIn(node_or_slice, (0, 1))
        test_self.assertSetEqual(value, {list})
        test_self.assertSetEqual(slice_, {int})
        if node_or_slice == 0:
          return {float}
        else:
          return {str}

    def test_fn(t):
      a, b = t
      return a, b

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[1].value, ((float, str),))
    self.assertTypes(fn_body[1].value.elts[0], float)
    self.assertTypes(fn_body[1].value.elts[1], str)

  def test_compare(self):
    # [FUNCTION DOC]
    # Scenario: two integer-typed arguments are compared. Expected: comparison resolution gets
    # the left type set and a list of right type sets, its boolean result types the whole
    # expression, and operands keep their integer types.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver typing arguments as integer and comparisons as boolean.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns the integer type for every argument.
        return {int}

      def res_compare(self, ns, types_ns, node, left, right):
        # [FUNCTION DOC]
        # Asserts left is the integer set and right is a list with one integer set; returns the
        # boolean type.
        test_self.assertSetEqual(left, {int})
        test_self.assertListEqual(right, [{int}])
        return {bool}

    def test_fn(a, b):
      return a < b

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, bool)
    self.assertTypes(fn_body[0].value.left, int)
    self.assertTypes(fn_body[0].value.comparators[0], int)

  def test_binop(self):
    # [FUNCTION DOC]
    # Scenario: two list-typed arguments combined with the matrix multiplication operator.
    # Expected: binary operation resolution receives both operand type sets and its float result
    # types the expression; operands keep the list type.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver typing arguments as list and the binary operation as float.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns the list type for every argument.
        return {list}

      def res_binop(self, ns, types_ns, node, left, right):
        # [FUNCTION DOC]
        # Asserts both operand type sets are list; returns the float type.
        test_self.assertSetEqual(left, {list})
        test_self.assertSetEqual(right, {list})
        return {float}

    def test_fn(a, b):
      return a @ b

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, float)
    self.assertTypes(fn_body[0].value.left, list)
    self.assertTypes(fn_body[0].value.right, list)

  def test_unop(self):
    # [FUNCTION DOC]
    # Scenario: a list-typed argument negated with the unary minus. Expected: unary operation
    # resolution yields float for the expression while the operand keeps the list type.

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver typing arguments as list and unary operations as float.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns the list type for every argument.
        return {list}

      def res_unop(self, ns, types_ns, node, opnd):
        # [FUNCTION DOC]
        # Returns the float type regardless of the operand.
        return {float}

    def test_fn(a):
      return -a

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, float)
    self.assertTypes(fn_body[0].value.operand, list)

  def test_tuple_literal(self):
    # [FUNCTION DOC]
    # Scenario: returning a tuple literal of two integer-typed arguments. Expected: the tuple is
    # typed as a tuple of two integers and each element is typed integer.

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver typing every argument as integer.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns the integer type for every argument.
        return {int}

    def test_fn(a, b):
      return a, b

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, ((int, int),))
    self.assertTypes(fn_body[0].value.elts[0], int)
    self.assertTypes(fn_body[0].value.elts[1], int)

  def test_list_literal(self):
    # [FUNCTION DOC]
    # Scenario: returning a list literal of two integer-typed arguments. Expected: the list
    # literal resolution receives the element type sets and, combining them, produces a
    # parameterized list-of-integer type for the literal; elements keep the integer type.

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver typing arguments as integer and list literals as lists of the union of element
      # types.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns the integer type for every argument.
        return {int}

      def res_list_literal(self, ns, elt_types):
        # [FUNCTION DOC]
        # Unions all element type sets and returns a set of list types, one per distinct element
        # type.
        all_types = set()
        for s in elt_types:
          all_types |= s
        return {List[t] for t in all_types}

    def test_fn(a, b):
      return [a, b]

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[0].value, List[int])
    self.assertTypes(fn_body[0].value.elts[0], int)
    self.assertTypes(fn_body[0].value.elts[1], int)

  def test_tuple_unpacking_syntactic(self):
    # [FUNCTION DOC]
    # Scenario: unpacking a tuple built syntactically from an integer and a float argument.
    # Expected: each target gets the type of its matching element (integer, float) via position
    # lookup, and the returned tuple is typed as a tuple of integer and float.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver typing argument a as integer and others as float, with tuple element slicing.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns integer for argument a, float for any other argument.
        if name == qual_names.QN('a'):
          return {int}
        else:
          return {float}

      def res_value(self, ns, value):
        # [FUNCTION DOC]
        # Asserts the value is a position zero or one; returns the bare integer type (not a set).
        test_self.assertIn(value, (0, 1))
        return int

      def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
        # [FUNCTION DOC]
        # Asserts position is zero or one, the indexed type is the integer-float tuple and the
        # index type is integer; returns the set of the element types at that position.
        test_self.assertIn(node_or_slice, (0, 1))
        test_self.assertSetEqual(value, {(int, float)})
        test_self.assertEqual(slice_, int)
        return {t[node_or_slice] for t in value}

    def test_fn(a, b):
      c, d = a, b
      return c, d

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[1].value, ((int, float),))
    self.assertTypes(fn_body[1].value.elts[0], int)
    self.assertTypes(fn_body[1].value.elts[1], float)

  def test_tuple_unpacking_operational(self):
    # [FUNCTION DOC]
    # Scenario: unpacking an argument whose type is a tuple of integer and float. Expected: the
    # targets get integer and float respectively via position lookups, and the returned tuple is
    # typed as a tuple of integer and float.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver typing the argument as an integer-float tuple, with tuple element slicing.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns a set containing the integer-float tuple type for every argument.
        return {(int, float)}

      def res_value(self, ns, value):
        # [FUNCTION DOC]
        # Asserts the value is a position zero or one; returns the bare integer type.
        test_self.assertIn(value, (0, 1))
        return int

      def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
        # [FUNCTION DOC]
        # Asserts position is zero or one, the indexed type is the integer-float tuple and the
        # index type is integer; returns the set of element types at that position.
        test_self.assertIn(node_or_slice, (0, 1))
        test_self.assertSetEqual(value, {(int, float)})
        test_self.assertEqual(slice_, int)
        return {t[node_or_slice] for t in value}

    def test_fn(a):
      c, d = a
      return c, d

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    self.assertTypes(fn_body[1].value, ((int, float),))
    self.assertTypes(fn_body[1].value.elts[0], int)
    self.assertTypes(fn_body[1].value.elts[1], float)

  def test_list_expansion_syntactic(self):
    # [FUNCTION DOC]
    # Scenario: unpacking into a list pattern from a tuple built syntactically from an integer
    # and a float argument. Expected: same as tuple unpacking: targets are integer and float and
    # the returned value is a tuple of integer and float.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver typing argument a as integer and others as float, with tuple element slicing.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns integer for argument a, float for any other argument.
        if name == qual_names.QN('a'):
          return {int}
        else:
          return {float}

      def res_value(self, ns, value):
        # [FUNCTION DOC]
        # Asserts the value is a position zero or one; returns the bare integer type.
        test_self.assertIn(value, (0, 1))
        return int

      def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
        # [FUNCTION DOC]
        # Asserts position is zero or one, the indexed type is the integer-float tuple and the
        # index type is integer; returns the set of element types at that position.
        test_self.assertIn(node_or_slice, (0, 1))
        test_self.assertSetEqual(value, {(int, float)})
        test_self.assertEqual(slice_, int)
        return {t[node_or_slice] for t in value}

    def test_fn(a, b):
      [c, d] = a, b
      return c, d

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    # TODO(mdan): Whether it's List or Tuple might be open for interpretation.
    self.assertTypes(fn_body[1].value, ((int, float),))
    self.assertTypes(fn_body[1].value.elts[0], int)
    self.assertTypes(fn_body[1].value.elts[1], float)

  def test_list_expansion_operational(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says "operational" but the body is identical to the syntactic
    # variant (it assigns from a tuple of two arguments rather than unpacking a single typed
    # argument), so it duplicates test_list_expansion_syntactic.
    # Scenario: unpacking into a list pattern from a tuple of an integer and a float argument.
    # Expected: targets are integer and float and the returned value is a tuple of both.

    test_self = self

    class Resolver(type_inference.Resolver):
      # [CLASS DOC]
      # Resolver typing argument a as integer and others as float, with tuple element slicing.

      def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        # [FUNCTION DOC]
        # Returns integer for argument a, float for any other argument.
        if name == qual_names.QN('a'):
          return {int}
        else:
          return {float}

      def res_value(self, ns, value):
        # [FUNCTION DOC]
        # Asserts the value is a position zero or one; returns the bare integer type.
        test_self.assertIn(value, (0, 1))
        return int

      def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
        # [FUNCTION DOC]
        # Asserts position is zero or one, the indexed type is the integer-float tuple and the
        # index type is integer; returns the set of element types at that position.
        test_self.assertIn(node_or_slice, (0, 1))
        test_self.assertSetEqual(value, {(int, float)})
        test_self.assertEqual(slice_, int)
        return {t[node_or_slice] for t in value}

    def test_fn(a, b):
      [c, d] = a, b
      return c, d

    node, _ = TestTranspiler(Resolver).transform(test_fn, None)
    fn_body = node.body

    # TODO(mdan): Whether it's List or Tuple might be open for interpretation.
    self.assertTypes(fn_body[1].value, ((int, float),))
    self.assertTypes(fn_body[1].value.elts[0], int)
    self.assertTypes(fn_body[1].value.elts[1], float)


if __name__ == '__main__':
  test.main()
