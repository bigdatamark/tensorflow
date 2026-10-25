# [MODULE DOC]
# Unit tests for the Module base class: name and name scope handling (construction inside
# scopes, graph numbering vs eager, memoization of the scope in TF2, wrapped methods, properties,
# overridden scopes, error handling), variable and submodule tracking (including distributed and
# composite variables), forward methods wrapped as compiled functions, abstract base classes,
# and the flatten helper (ordering key, ignored attributes, paths, cycles, errors). Also defines
# small helper module classes and functions used as fixtures. No import-time side effects other
# than defining classes and tests.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.Module`."""

import abc
import collections
import itertools
import sys
import unittest

from absl.testing import parameterized

from tensorflow.python import tf2
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as distributed_values
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.module import module
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class TestModuleNaming(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests module naming and name scope behavior: explicit names, scopes entered by wrapped
  # methods, properties, overrides, name validation, numbering and error cleanup.

  def test_single_name(self):
    # [FUNCTION DOC]
    # A module created with an explicit name reports that name and a scope name with a trailing
    # slash.
    mod = module.Module(name="simple")
    self.assertEqual(mod.name, "simple")
    self.assertEqual(mod.name_scope.name, "simple/")

  def test_construct_in_scope(self):
    # [FUNCTION DOC]
    # A module created inside an outer name scope keeps its short name but its scope name is
    # nested under the outer scope.
    with ops.name_scope("foo", skip_on_eager=False):
      mod = module.Module(name="bar")
    self.assertEqual(mod.name, "bar")
    self.assertEqual(mod.name_scope.name, "foo/bar/")

  def test_enters_name_scope_in_call(self):
    # [FUNCTION DOC]
    # Calling a module whose call method is wrapped returns the module scope name, on each of
    # three repeated calls.
    mod = ReturnsNameScopeModule()
    for _ in range(3):
      self.assertEqual(mod(), mod.name_scope.name)

  def test_enters_name_scope_in_other_method(self):
    # [FUNCTION DOC]
    # Same as the call test but for a different wrapped method; the module scope is entered
    # on each of three calls.
    mod = ReturnsNameScopeModule()
    for _ in range(3):
      self.assertEqual(mod.alternative_forward(), mod.name_scope.name)

  def test_subclassed_module(self):
    # [FUNCTION DOC]
    # Wrapped methods defined on both a base module class and its subclass both enter the
    # instance's scope.
    mod = SubclassedReturnsNameScopeModule()
    for _ in range(3):
      self.assertEqual(mod.alternative_forward(), mod.name_scope.name)
      self.assertEqual(mod.alternative_alternative_forward(),
                       mod.name_scope.name)

  def test_submodule_created_late(self):
    # [FUNCTION DOC]
    # A leaf module created later inside a wrapped method gets the same default name but its
    # scope is nested under the parent scope.
    m = TreeModule()
    self.assertEqual(m.name, "tree_module")
    self.assertEqual(m.name_scope.name, "tree_module/")
    leaf1 = m.new_leaf()
    self.assertEqual(leaf1.name, "tree_module")
    self.assertEqual(leaf1.name_scope.name, "tree_module/tree_module/")

  def test_does_not_evaluate_property_methods(self):
    # [FUNCTION DOC]
    # Constructing a module with a property that raises must not evaluate the property; only
    # explicit access raises an assertion error.
    mod = PropertyThrowsWhenCalledModule()
    with self.assertRaises(AssertionError):
      mod.raise_assertion_error  # pylint: disable=pointless-statement

  def test_overridden_name_scope(self):
    # [FUNCTION DOC]
    # A subclass overriding the scope property: wrapped methods report the overridden scope name.
    mod = ModuleOverridingNameScope()
    self.assertEqual(mod(), mod.name_scope.name)
    self.assertEqual(mod.alternative_forward(), mod.name_scope.name)

  def test_patched_callable(self):
    # [FUNCTION DOC]
    # A plain function assigned as an instance attribute is not a method, so the module scope is
    # not entered and the current scope name is empty.
    with ops.name_scope("foo", skip_on_eager=False):
      mod = module.Module(name="bar")
    mod.foo = get_name_scope
    # `foo` is not a method so we do not re-enter the name scope.
    self.assertEqual(mod.foo(), "")

  def test_property(self):
    # [FUNCTION DOC]
    # A property whose getter and setter are wrapped sees the module scope in both; both captured
    # scope names equal the module's scope.
    mod = PropertyModule()
    mod.some_property = None, None  # None, None for the linter.
    getter_scope_name, setter_scope_name = mod.some_property
    self.assertEqual(getter_scope_name, "property_module/")
    self.assertEqual(setter_scope_name, "property_module/")

  def test_property_no_name_scope(self):
    # [FUNCTION DOC]
    # A property with unwrapped getter and setter sees an empty scope name in both.
    mod = PropertyModule()
    mod.no_name_scope_property = None, None  # None, None for the linter.
    getter_scope_name, setter_scope_name = mod.no_name_scope_property
    self.assertEqual(getter_scope_name, "")
    self.assertEqual(setter_scope_name, "")

  def test_invalid_name(self):
    # [FUNCTION DOC]
    # Creating a module with a name that is not a valid identifier raises a value error with the
    # message that it is not a valid module name.
    msg = ".* is not a valid module name"
    with self.assertRaisesRegex(ValueError, msg):
      module.Module(name="$Foo")

  @test_util.run_in_graph_and_eager_modes
  def test_modules_not_numbered_in_eager(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes but skips unless eager. Creating the same nested recursive
    # module twice gives identical (not numbered) scope names both times.
    if not context.executing_eagerly():
      self.skipTest("Eager specific")

    mod = RecursiveModule(2)
    self.assertEqual(mod.name_scope.name, "badger/")
    self.assertEqual(mod.child.name_scope.name, "badger/badger/")

    mod = RecursiveModule(2)
    self.assertEqual(mod.name_scope.name, "badger/")
    self.assertEqual(mod.child.name_scope.name, "badger/badger/")

  @test_util.run_in_graph_and_eager_modes
  def test_module_numbering_in_graph(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes but skips unless in graph mode. Creating the same recursive
    # module twice makes the second outer scope numbered with a suffix while the child scope
    # name stays unnumbered under it.
    if context.executing_eagerly():
      self.skipTest("Graph specific")

    mod = RecursiveModule(2)
    self.assertEqual(mod.name_scope.name, "badger/")
    self.assertEqual(mod.child.name_scope.name, "badger/badger/")

    mod = RecursiveModule(2)
    self.assertEqual(mod.name_scope.name, "badger_1/")
    self.assertEqual(mod.child.name_scope.name, "badger_1/badger/")

  def test_ctor_error_closes_name_scope(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment refers to a metaclass that closes the scope, but the module
    # class in this repository has no such metaclass; the scope is closed by the context manager
    # in the constructor.
    # A constructor that calls the base constructor and then raises must leave no open scope.
    with self.assertRaises(ErrorModuleError):
      # If super constructor is called then a name scope is opened then an error
      # is thrown. The metaclass should handle this and close the namescope
      # before re-throwing the exception.
      ErrorModule(call_super=True)

    self.assertEqual("", get_name_scope())

  def test_ctor_error_handles_ctor_not_opening_name_scope(self):
    # [FUNCTION DOC]
    # A constructor that raises without calling the base constructor must propagate its own
    # error, with no scope left open.
    with self.assertRaises(ErrorModuleError):
      # If super ctor is not called then the name scope isn't opened. We need to
      # ensure that this doesn't trigger an exception (e.g. the metaclass trying
      # to __exit__ a non-existent name scope).
      ErrorModule(call_super=False)

    self.assertEqual("", get_name_scope())

  def test_forward_method_closes_name_scope(self):
    # [FUNCTION DOC]
    # A call that raises an error must propagate it and leave no module scope open.
    mod = ErrorModule(call_super=True, raise_in_constructor=False)
    with self.assertRaises(ErrorModuleError):
      mod()

    self.assertEqual("", get_name_scope())

  def test_get_attr_doesnt_enter_name_scope(self):
    # [FUNCTION DOC]
    # Defines a module recording the current scope name in its missing-attribute hook. Accessing
    # a missing attribute raises an attribute error, and the recorded scope name is empty.
    scope_names = []

    class GetAttrModule(module.Module):
      # [CLASS DOC]
      # Local module subclass whose missing-attribute hook records the attribute name and the
      # current scope name in the enclosing list, then defers to the parent.

      def __getattr__(self, name):
        # [FUNCTION DOC]
        # Records the requested name with the current scope name, then defers to the parent
        # hook (which raises an attribute error for unknown names).
        scope_names.append((name, get_name_scope()))
        return super().__getattr__(name)

    mod = GetAttrModule()
    with self.assertRaises(AttributeError):
      mod.does_not_exist  # pylint: disable=pointless-statement
    self.assertIn(("does_not_exist", ""), scope_names)

  def test_get_attribute_doesnt_enter_name_scope(self):
    # [FUNCTION DOC]
    # Like the missing-attribute test, but the module records scope names in the hook that runs on
    # every attribute access; a missing attribute raises an error and the recorded scope is empty.
    scope_names = []

    class GetAttributeModule(module.Module):
      # [CLASS DOC]
      # Local module subclass whose attribute access hook records each accessed name and the
      # current scope name in the enclosing list, then defers to the parent.

      def __getattribute__(self, name):
        # [FUNCTION DOC]
        # Records the accessed attribute name with the current scope name, then returns the
        # parent lookup result (raising for unknown names).
        scope_names.append((name, get_name_scope()))
        return super().__getattribute__(name)

    mod = GetAttributeModule()
    with self.assertRaises(AttributeError):
      mod.does_not_exist  # pylint: disable=pointless-statement
    self.assertIn(("does_not_exist", ""), scope_names)


class VariableNamingTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that variables created inside a module scope get names prefixed by the scope chain.

  def test_variable_names(self):
    # [FUNCTION DOC]
    # Builds a three-deep recursive module and asserts each level's variable name includes the
    # scopes of all enclosing modules.
    mod = RecursiveModule(3)
    self.assertEqual(mod.w.name, "badger/mushroom:0")
    self.assertEqual(mod.child.w.name, "badger/badger/mushroom:0")
    self.assertEqual(mod.child.child.w.name, "badger/badger/badger/mushroom:0")


class NameScopeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests whether the module scope object is memoized, depending on TF1 or TF2 behavior.

  @test_util.run_deprecated_v1
  def test_not_memoized_in_tf1(self):
    # [FUNCTION DOC]
    # Skipped under TF2 behavior. Under TF1, two reads of the scope property give distinct
    # objects with equal names.
    if tf2.enabled():
      self.skipTest("Requires TF1")

    mod = module.Module(name="name")
    name_scope_1 = mod.name_scope
    name_scope_2 = mod.name_scope
    self.assertIsNot(name_scope_1, name_scope_2)
    self.assertEqual(name_scope_1.name, name_scope_2.name)

  def test_memoized_in_tf2(self):
    # [FUNCTION DOC]
    # Skipped unless TF2 behavior is enabled. Two reads of the scope property return the very
    # same object.
    if not tf2.enabled():
      self.skipTest("Requires TF2")

    mod = module.Module(name="name")
    name_scope_1 = mod.name_scope
    name_scope_2 = mod.name_scope
    self.assertIs(name_scope_1, name_scope_2)


class VariableTrackingTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests variable discovery on modules: ordering, trainable filtering, distributed variable
  # types and composite tensors holding variables.

  def test_variables(self):
    # [FUNCTION DOC]
    # For a three-level recursive module, each level's variables are its own followed by those of
    # its descendants.
    m = RecursiveModule(3)
    self.assertEqual(m.variables, (m.w, m.child.w, m.child.child.w))
    self.assertEqual(m.child.variables, (m.child.w, m.child.child.w))
    self.assertEqual(m.child.child.variables, (m.child.child.w,))

  def test_trainable_variables(self):
    # [FUNCTION DOC]
    # Same as the variables test but for trainable variables, all of which are trainable here.
    m = RecursiveModule(3)
    self.assertEqual(m.trainable_variables,
                     (m.w, m.child.w, m.child.child.w))
    self.assertEqual(m.child.trainable_variables,
                     (m.child.w, m.child.child.w))
    self.assertEqual(m.child.child.trainable_variables, (m.child.child.w,))

  def test_trainable_variables_ignores_non_trainable(self):
    # [FUNCTION DOC]
    # With non-trainable variables at every level, each level's trainable list is empty.
    m = RecursiveModule(3, trainable=False)
    self.assertEqual(len(m.trainable_variables), 0)
    self.assertEqual(len(m.child.trainable_variables), 0)
    self.assertEqual(len(m.child.child.trainable_variables), 0)

  def test_supports_distributed_variables(self):
    # [FUNCTION DOC]
    # Assigns mirrored, TPU mirrored and aggregating distributed variable wrappers to module
    # attributes and asserts they are all found, in attribute order.
    mirrored = distributed_values.MirroredVariable(
        None, [variables.Variable(1.)], variables.VariableAggregation.SUM)
    tpu = tpu_values.TPUMirroredVariable(
        strategy=None, values=[variables.Variable(42.)], aggregation=None)
    aggregating = ps_values.AggregatingVariable(
        strategy=None, v=variables.Variable(1.), aggregation=None)

    m = module.Module()
    m.a = mirrored
    m.b = tpu
    m.c = aggregating
    self.assertEqual(m.variables, (mirrored, tpu, aggregating))

  def test_composite_variable(self):
    # [FUNCTION DOC]
    # Defines a composite tensor type whose components are variables and stores an instance on a
    # module. Asserts the module's variables are the composite's component variables.

    class Spec(type_spec.TypeSpec):
      # [CLASS DOC]
      # Minimal type specification for the local composite variable class; only the value type
      # and component conversion are implemented, the rest are stubs.

      value_type = property(lambda self: CompositeVariable)

      def _component_specs(self):
        # [FUNCTION DOC]
        # Stub returning nothing; component specs are not needed by this test.
        pass

      def _serialize(self):
        # [FUNCTION DOC]
        # Stub returning nothing; serialization is not needed by this test.
        pass

      def _to_components(self, value):
        # [FUNCTION DOC]
        # Returns the list of variables stored in the given composite value.
        return value._variables

      def _from_components(self, variable_list):
        # [FUNCTION DOC]
        # Rebuilds a composite variable from a list of component variables.
        return CompositeVariable(variable_list)

    class CompositeVariable(composite_tensor.CompositeTensor):
      # [CLASS DOC]
      # Local composite tensor that simply wraps a list of variables.

      def __init__(self, variable_list):
        # [FUNCTION DOC]
        # Stores the given list of variables.
        self._variables = variable_list

      @property
      def _type_spec(self):
        # [FUNCTION DOC]
        # Property returning a new instance of the local type specification.
        return Spec()

    m = module.Module()
    m.a = CompositeVariable([variables.Variable(1.), variables.Variable(2.)])
    self.assertEqual(list(m.variables), list(m.a._variables))


class ModuleTrackingTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests discovery of submodules, including ones created after construction.

  def test_submodules(self):
    # [FUNCTION DOC]
    # For a three-level recursive module, each level's submodules are all its descendants in
    # order, and the deepest has none.
    m = RecursiveModule(3)
    self.assertEqual(list(m.submodules), [m.child, m.child.child])
    self.assertEqual(list(m.child.submodules), [m.child.child])
    self.assertEqual(list(m.child.child.submodules), [])

  def test_non_ctor_submodule(self):
    # [FUNCTION DOC]
    # Leaves added after construction are picked up: the submodule set grows as each is created.
    m = TreeModule()
    leaf1 = m.new_leaf()
    self.assertEqual(set(m.submodules), {leaf1})
    leaf2 = m.new_leaf()
    self.assertEqual(set(m.submodules), {leaf1, leaf2})


class ForwardMethodsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that compiled functions wrapping name-scoped methods keep the module scope.

  def testFunctionType(self):
    # [FUNCTION DOC]
    # The decorated forward methods (with and without autograph) are function wrapper objects.
    mod = ModuleWithFunctionAnnotatedCall()
    self.assertIsInstance(mod.forward, def_function.Function)
    self.assertIsInstance(mod.forward_ag, def_function.Function)

  def testEntersNameScope_call(self):
    # [FUNCTION DOC]
    # Calling each compiled forward method and evaluating the result gives the module scope name
    # as bytes.
    mod = ModuleWithFunctionAnnotatedCall()
    self.assertEqual(self.evaluate(mod.forward()),
                     b"module_with_function_annotated_call/")
    self.assertEqual(self.evaluate(mod.forward_ag()),
                     b"module_with_function_annotated_call/")

  def testEntersNameScope_concreteFunction(self):
    # [FUNCTION DOC]
    # Same check as the call test, but through concrete functions obtained from the compiled
    # methods.
    mod = ModuleWithFunctionAnnotatedCall()
    self.assertEqual(self.evaluate(mod.forward.get_concrete_function()()),
                     b"module_with_function_annotated_call/")
    self.assertEqual(self.evaluate(mod.forward_ag.get_concrete_function()()),
                     b"module_with_function_annotated_call/")


class AbcTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that modules work with abstract base class machinery.

  def testAbstract(self):
    # [FUNCTION DOC]
    # Instantiating a module with an unimplemented abstract method raises a type error saying it
    # cannot be instantiated.
    msg = "Can't instantiate.*abstract"
    with self.assertRaisesRegex(TypeError, msg):
      AbstractModule()  # pylint: disable=abstract-class-instantiated

  def testConcrete(self):
    # [FUNCTION DOC]
    # A concrete subclass can be called: squares its input, reports its own scope name, and the
    # scope is closed afterwards.
    mod = ConcreteModule()
    x, scope_name = mod(2.)
    self.assertEqual(x, 4.)
    self.assertEqual(scope_name, "concrete_module/")
    self.assertEqual(get_name_scope(), "")


def get_name_scope():
  # [FUNCTION DOC]
  # HELPER: test utility returning the current name scope. Opens a temporary scope, strips the
  # last two path components of its name (the temporary one) and returns the remainder with a
  # trailing slash, or an empty string when no scope is active.
  with ops.name_scope("x", skip_on_eager=False) as ns:
    ns = "/".join(ns.split("/")[:-2])
    return ns + "/" if ns else ""


class ErrorModuleError(Exception):
  # [CLASS DOC]
  # HELPER: exception type raised deliberately by the error fixture module.
  pass


class ErrorModule(module.Module):
  # [CLASS DOC]
  # HELPER: fixture module that raises a deliberate error either in its constructor or when
  # called, optionally without calling the base constructor.

  def __init__(self, call_super, raise_in_constructor=True):
    # [FUNCTION DOC]
    # Calls the base constructor only if call_super is true, then raises the deliberate error if
    # raise_in_constructor (default true) is set.
    if call_super:
      super().__init__()
    if raise_in_constructor:
      raise ErrorModuleError("Deliberate error!")

  def __call__(self):
    # [FUNCTION DOC]
    # Always raises the deliberate error.
    raise ErrorModuleError("Deliberate error!")


class RecursiveModule(module.Module):
  # [CLASS DOC]
  # HELPER: fixture module named "badger" holding a variable and, if depth allows, a child of the
  # same type with depth reduced by one.

  def __init__(self, depth, trainable=True):
    # [FUNCTION DOC]
    # Inputs: depth (number of levels including this one) and trainable (whether the variable is
    # trainable, default true). Inside the module scope, creates the child (or nothing at depth
    # one) and a scalar variable named mushroom.
    super().__init__(name="badger")
    with self.name_scope:
      self.child = None
      if depth > 1:
        self.child = RecursiveModule(depth - 1, trainable=trainable)
      self.w = variables.Variable(1.0, trainable=trainable, name="mushroom")


class AbstractModule(module.Module, metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # HELPER: fixture module using the abstract base class metaclass, with an abstract call.

  @abc.abstractmethod
  def __call__(self, x):
    # [FUNCTION DOC]
    # Abstract call method; does nothing and must be overridden.
    pass


class ConcreteModule(AbstractModule):
  # [CLASS DOC]
  # HELPER: concrete implementation of the abstract fixture with a scope-wrapped call.

  @module.Module.with_name_scope
  def __call__(self, x):
    # [FUNCTION DOC]
    # Returns a pair: the input squared and the current scope name.
    return x ** 2, get_name_scope()


class TreeModule(module.Module):
  # [CLASS DOC]
  # HELPER: fixture module that can grow child modules of its own type after construction.

  def __init__(self, name=None):
    # [FUNCTION DOC]
    # Optional name is passed to the base constructor; starts with an empty list of leaves.
    super().__init__(name=name)
    self._leaves = []

  @module.Module.with_name_scope
  def new_leaf(self, name=None):
    # [FUNCTION DOC]
    # Creates a child of the same type (optional name), inside this module's scope, remembers it
    # in the leaf list and returns it.
    leaf = TreeModule(name=name)
    self._leaves.append(leaf)
    return leaf


class ReturnsNameScopeModule(module.Module):
  # [CLASS DOC]
  # HELPER: fixture module whose wrapped methods return the scope name they run in.

  @module.Module.with_name_scope
  def alternative_forward(self):
    # [FUNCTION DOC]
    # Returns the current scope name seen inside the wrapped method.
    return get_name_scope()

  @module.Module.with_name_scope
  def __call__(self):
    # [FUNCTION DOC]
    # Returns the current scope name seen inside the wrapped call.
    return get_name_scope()


class SubclassedReturnsNameScopeModule(ReturnsNameScopeModule):
  # [CLASS DOC]
  # HELPER: subclass fixture adding one more wrapped method.

  @module.Module.with_name_scope
  def alternative_alternative_forward(self):
    # [FUNCTION DOC]
    # Returns the current scope name seen inside the wrapped method.
    return get_name_scope()


class PropertyThrowsWhenCalledModule(module.Module):
  # [CLASS DOC]
  # HELPER: fixture module with a property that raises an assertion error when read.

  @property
  def raise_assertion_error(self):
    # [FUNCTION DOC]
    # Property that always raises an assertion error.
    raise AssertionError


class ModuleOverridingNameScope(ReturnsNameScopeModule):
  # [CLASS DOC]
  # HELPER: fixture that overrides the scope property to a fixed scope named yolo.

  @property
  def name_scope(self):
    # [FUNCTION DOC]
    # Property returning a new scope context for the fixed name yolo, ignoring the module name.
    return ops.name_scope("yolo/", skip_on_eager=False)


class ModuleWithFunctionAnnotatedCall(module.Module):
  # [CLASS DOC]
  # HELPER: fixture module with two compiled, scope-wrapped methods (without and with autograph).

  @def_function.function(autograph=False)
  @module.Module.with_name_scope
  def forward(self):
    # [FUNCTION DOC]
    # Returns the current scope name as seen during tracing without autograph.
    return get_name_scope()

  @def_function.function(autograph=True)
  @module.Module.with_name_scope
  def forward_ag(self):
    # [FUNCTION DOC]
    # Returns the current scope name as seen during tracing with autograph.
    return get_name_scope()


class PropertyModule(module.Module):
  # [CLASS DOC]
  # HELPER: fixture module with one property whose getter and setter run in the module scope and
  # one whose accessors do not. Each setter stores the scope name it saw; each getter returns the
  # scope name it saw together with the stored setter scope name.

  def __init__(self):
    # [FUNCTION DOC]
    # Calls the base constructor and initializes the recorded setter scope name to nothing.
    super().__init__()
    self._setter_scope_name = None

  @property
  @module.Module.with_name_scope
  def some_property(self):
    # [FUNCTION DOC]
    # Scoped getter: returns a pair of the scope name seen now and the scope name recorded by the
    # setter.
    getter_scope_name = get_name_scope()
    return getter_scope_name, self._setter_scope_name

  @some_property.setter
  @module.Module.with_name_scope
  def some_property(self, my_property):
    # [FUNCTION DOC]
    # Scoped setter: ignores the value and records the current scope name.
    self._setter_scope_name = get_name_scope()

  @property
  def no_name_scope_property(self):
    # [FUNCTION DOC]
    # Unscoped getter: returns a pair of the scope name seen now and the recorded setter scope
    # name.
    getter_scope_name = get_name_scope()
    return getter_scope_name, self._setter_scope_name

  @no_name_scope_property.setter
  def no_name_scope_property(self, my_property):
    # [FUNCTION DOC]
    # Unscoped setter: ignores the value and records the current scope name.
    self._setter_scope_name = get_name_scope()

NamedPair = collections.namedtuple("NamedPair", ("first", "second"))
mk_index_dict = lambda v: dict(enumerate(v))


class FlattenTest(parameterized.TestCase, test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Parameterized tests of the module flatten routine: container types, traversal order, ignored
  # attributes, path output, reference cycles and error reporting.

  @parameterized.parameters(lambda v: NamedPair(*v), list, tuple, mk_index_dict)
  def test_flatten(self, container_type):
    # [FUNCTION DOC]
    # Parameterized over four container kinds (named pair, list, tuple, index mapping). Builds a
    # parent with one child; flattening non-recursively with a member-type predicate yields only
    # the parent's members, and recursively yields the parent's then the child's.
    parent = SimpleModule(container_type=container_type)
    child = parent.c

    self.assertEqual(
        list(parent._flatten(recursive=False, predicate=is_member)),
        [parent.a[0], parent.a[1], parent.z])

    self.assertEqual(
        list(parent._flatten(predicate=is_member)),
        [parent.a[0], parent.a[1], parent.z, child.a[0], child.a[1], child.z])

  def test_attribute_traversal_key(self):
    # [FUNCTION DOC]
    # A module that orders attribute traversal via a custom key returns its variables in that
    # order: trainable, then non-trainable, then the extra variable.
    mod = LayerModule()
    self.assertEqual(
        mod.variables,
        mod._trainable_variables + mod._non_trainable_variables + [mod._bonus])

  def test_attributes_to_ignore(self):
    # [FUNCTION DOC]
    # A subclass extends the ignored attribute set with a submodule and a variable. After setting
    # those and one normal variable, there are no submodules and only the normal variable is
    # found.
    class DangerousModule(module.Module):
      # [CLASS DOC]
      # Local module subclass whose ignore set adds two attribute names to the inherited ones.
      _TF_MODULE_IGNORED_PROPERTIES = frozenset(itertools.chain(
          ("dangerous_submodule", "dangerous_variable"),
          module.Module._TF_MODULE_IGNORED_PROPERTIES
      ))

    mod = DangerousModule()
    mod.dangerous_submodule = module.Module()
    mod.dangerous_variable = variables.Variable(1.)
    mod.normal_variable = variables.Variable(2.)

    self.assertEmpty(mod.submodules)
    self.assertLen(mod.variables, 1)
    self.assertEqual(mod.variables[0], mod.normal_variable)

  @unittest.skipIf(sys.version_info.major == 3 and sys.version_info.minor == 12,
                   reason="b/313658911: _TupleWrapper __dict__ attribute error")
  def test_with_path(self):
    # [FUNCTION DOC]
    # Skipped on Python 3.12 (known tuple wrapper attribute error). Builds a module with a shared
    # variable reachable through nested containers and two attributes pointing to the same child.
    # Flattening with paths yields a mapping of full attribute paths to variables, listing the
    # shared child's variables once under each attribute name.
    mod = module.Module()
    mod.w = variables.Variable(1.)
    mod.encoder = module.Module()
    mod.encoder.w = [({"k": mod.w}, {"k": mod.w})]
    mod.decoder = mod.encoder

    state_dict = dict(
        mod._flatten(with_path=True, predicate=module._is_variable))

    self.assertEqual(state_dict,
                     {("w",): mod.w,
                      ("encoder", "w", 0, 0, "k"): mod.encoder.w[0][0]["k"],
                      ("encoder", "w", 0, 1, "k"): mod.encoder.w[0][1]["k"],
                      ("decoder", "w", 0, 0, "k"): mod.decoder.w[0][0]["k"],
                      ("decoder", "w", 0, 1, "k"): mod.decoder.w[0][1]["k"]},)

  def test_cycles_with_path(self):
    # [FUNCTION DOC]
    # Like the path test, but the shared child refers back to the root, creating cycles.
    # Flattening with paths must terminate and yield the expected paths, including the variable
    # reached once through each back reference.
    mod = module.Module()
    mod.w = variables.Variable(1.)
    mod.encoder = module.Module()
    mod.encoder.w = [({"k": mod.w}, {"k": mod.w})]
    mod.decoder = mod.encoder

    # This introduces two cycles: on mod.encoder.mod and mod.decoder.mod.
    mod.decoder.mod = mod

    state_dict = dict(
        mod._flatten(with_path=True, predicate=module._is_variable))

    self.assertEqual(state_dict,
                     {("w",): mod.w,
                      ("encoder", "mod", "w"): mod.encoder.mod.w,
                      ("decoder", "mod", "w"): mod.decoder.mod.w,
                      ("encoder", "w", 0, 0, "k"): mod.encoder.w[0][0]["k"],
                      ("encoder", "w", 0, 1, "k"): mod.encoder.w[0][1]["k"],
                      ("decoder", "w", 0, 0, "k"): mod.decoder.w[0][0]["k"],
                      ("decoder", "w", 0, 1, "k"): mod.decoder.w[0][1]["k"]},)

  def test_raises_error_with_path(self):
    # [FUNCTION DOC]
    # A mapping attribute with keys that cannot be ordered makes flattening fail; the variables
    # property raises a value error naming the offending property.
    non_orderable = object

    m = module.Module()
    m.layers = {non_orderable(): None, non_orderable(): None}
    with self.assertRaisesRegex(ValueError,
                                "Error processing property 'layers'"):
      m.variables  # pylint: disable=pointless-statement


class LayerModule(module.Module):
  # [CLASS DOC]
  # HELPER: fixture module with lists of trainable and non-trainable variables plus one extra,
  # overriding the variables property to impose a custom attribute traversal order.

  def __init__(self):
    # [FUNCTION DOC]
    # Creates two trainable-named variables, two non-trainable-named variables and one extra
    # variable. Note all are created trainable; the names only reflect list groupings.
    super().__init__()
    self._trainable_variables = [
        variables.Variable(1., name="a"),
        variables.Variable(2., name="b"),
    ]
    self._non_trainable_variables = [
        variables.Variable(3., name="c"),
        variables.Variable(4., name="d"),
    ]
    self._bonus = variables.Variable(5., name="e")

  @property
  def variables(self):
    # [FUNCTION DOC]
    # Property returning a list (not a tuple) of variables, flattened with a custom traversal key
    # that visits the trainable group first, then non-trainable, then other attributes by name.
    def key_function(name):
      # [FUNCTION DOC]
      # Sort key for attribute names: group index (zero, one, or two for others) then the name.
      indexes = {"_trainable_variables": 0, "_non_trainable_variables": 1}
      return indexes.get(name, 2), name

    return list(
        self._flatten(
            predicate=module._is_variable,
            attribute_traversal_key=key_function))


class MemberType:
  # [CLASS DOC]
  # HELPER: marker type with no behavior, searched for by the flatten tests.
  """A simple type to search for."""
  pass


class SimpleModule(module.Module):
  # [CLASS DOC]
  # HELPER: fixture module holding a marker object, a container of two marker objects and
  # optionally a child of the same type.

  def __init__(self, create_child=True, container_type=list):
    # [FUNCTION DOC]
    # Inputs: create_child (add one childless child of the same type, default true) and
    # container_type (callable used to wrap the two markers, default list). Sets the marker
    # attribute, the container attribute and optionally the child attribute.
    super().__init__()
    self.z = MemberType()
    self.a = container_type([MemberType(), MemberType()])
    if create_child:
      self.c = SimpleModule(create_child=False)

is_member = lambda v: isinstance(v, MemberType)

if __name__ == "__main__":
  test.main()
