# [MODULE DOC]
# Tests and benchmarks for the trace type builder (the function converting Python values into
# TraceTypes). Test groups: builder tests (equality, aliasing of variables, tensors versus specs,
# attrs, tuples, lists, dicts, wrapped named tuples, error handling, placeholder values), cast
# tests for the default types, legacy-signature tests, memory-leak tests and cache-key generation
# benchmarks. Defines small fixture classes (attrs-like class, generic class, a custom nest
# dataclass). Runs under the TensorFlow test runner with v2 behavior enabled.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests and benchmarks for the trace_type module."""

import collections
import dataclasses
import timeit
import weakref

from absl.testing import parameterized
import numpy as np

from tensorflow.core.function import trace_type
from tensorflow.core.function.trace_type import custom_nest_trace_type
from tensorflow.core.function.trace_type import default_types
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


class TestAttr:
  # [CLASS DOC]
  # HELPER: Fixture imitating one attrs attribute descriptor; only carries a name.
  """Helps test attrs collections."""

  def __init__(self, name):
    # [FUNCTION DOC]
    # Stores the attribute name.
    self.name = name


class TestAttrsClass:
  # [CLASS DOC]
  # HELPER: Fixture imitating an attrs-decorated class with attributes a and b (exposes the attrs
  # metadata list and stores two values). Equal when both attributes are equal.
  """Helps test attrs collections."""

  __attrs_attrs__ = (TestAttr('a'), TestAttr('b'))

  def __init__(self, a, b):
    # [FUNCTION DOC]
    # Stores the two attribute values a and b.
    self.a = a
    self.b = b

  def __eq__(self, other):
    # [FUNCTION DOC]
    # True if the other is also this class and both attributes are equal.
    return isinstance(
        other, TestAttrsClass) and self.a == other.a and self.b == other.b


class DummyGenericClass:
  # [CLASS DOC]
  # HELPER: Empty fixture class whose instances are generic weak-referenceable objects, used to
  # check the builder handles arbitrary objects without leaking memory.
  """Helps test memory leaks for GenericType."""


@dataclasses.dataclass
class MaskedTensor:
  # [CLASS DOC]
  # HELPER: Fixture data class implementing the custom nest protocol. Holds a boolean mask (kept
  # as metadata) and a tensor value (the single component). Used to test casting of custom nest
  # types.
  mask: bool
  value: tensor_lib.Tensor

  def __tf_flatten__(self):
    # [FUNCTION DOC]
    # Custom nest protocol flatten: returns a pair of metadata (a one-item tuple with the mask)
    # and components (a one-item tuple with the value).
    metadata = (self.mask,)
    components = (self.value,)
    return metadata, components

  @classmethod
  def __tf_unflatten__(cls, metadata, leaves):
    # [FUNCTION DOC]
    # Custom nest protocol unflatten (class method): rebuilds an instance from the metadata (mask
    # first item) and leaves (value first item).
    mask = metadata[0]
    value = leaves[0]
    return MaskedTensor(mask=mask, value=value)


class TraceTypeBuilderTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the builder function that turns values into TraceTypes (equality semantics,
  # aliasing, placeholder generation and error handling). Most tests run in graph and eager mode.

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testCompositeAndSpec(self):
    # [FUNCTION DOC]
    # Scenario: a ragged tensor and its matching spec. Expects both to produce equal TraceTypes.
    composite_tensor = ragged_tensor.RaggedTensor.from_row_splits(
        values=[1, 2, 3], row_splits=[0, 2, 3])
    spec = ragged_tensor.RaggedTensorSpec([2, None], dtypes.int32)

    self.assertEqual(
        trace_type.from_value(composite_tensor), trace_type.from_value(spec))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testVariableAliasing(self):
    # [FUNCTION DOC]
    # Scenario: tuples of three distinct variables versus the same variable repeated. Expects the
    # two TraceTypes to differ, and tuples built from other variables with the same aliasing
    # pattern to equal the corresponding earlier ones (identity pattern, not variable identity).
    v1 = resource_variable_ops.ResourceVariable([1])
    v2 = resource_variable_ops.ResourceVariable([1])
    v3 = resource_variable_ops.ResourceVariable([1])
    all_unique = trace_type.from_value((v1, v2, v3))
    all_same = trace_type.from_value((v1, v1, v1))
    self.assertNotEqual(all_unique, all_same)

    v3 = resource_variable_ops.ResourceVariable([2])
    v4 = resource_variable_ops.ResourceVariable([2])
    v5 = resource_variable_ops.ResourceVariable([2])
    all_unique_again = trace_type.from_value((v3, v4, v5))
    all_same_again = trace_type.from_value((v4, v4, v4))
    self.assertEqual(all_unique, all_unique_again)
    self.assertEqual(all_same, all_same_again)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTensorEquality(self):
    # [FUNCTION DOC]
    # Scenario: tracing types of tensors that differ in shape, dtype or value. Expects different
    # shapes or dtypes to give unequal types, and equal shape and dtype with different values to
    # give equal types.
    context = trace_type.InternalTracingContext()
    tensor_a = array_ops.zeros([11, 3, 5],
                               dtype=dtypes.int32).__tf_tracing_type__(context)
    tensor_b = array_ops.zeros([11, 4, 5],
                               dtype=dtypes.int32).__tf_tracing_type__(context)
    tensor_c = array_ops.zeros(
        [11, 3, 5], dtype=dtypes.float32).__tf_tracing_type__(context)
    tensor_d = array_ops.ones([11, 3, 5],
                              dtype=dtypes.int32).__tf_tracing_type__(context)

    self.assertNotEqual(tensor_a, tensor_b)
    self.assertNotEqual(tensor_a, tensor_c)
    self.assertNotEqual(tensor_b, tensor_c)
    self.assertEqual(tensor_a, tensor_d)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTensorAndSpecEquality(self):
    # [FUNCTION DOC]
    # Scenario: a tensor, a matching unnamed spec and a matching named spec. Expects the tensor to
    # equal the unnamed spec but not the named one.
    context = trace_type.InternalTracingContext()
    tensor = array_ops.zeros([11, 3, 5],
                             dtype=dtypes.int32).__tf_tracing_type__(context)
    spec = tensor_spec.TensorSpec(
        [11, 3, 5], dtype=dtypes.int32).__tf_tracing_type__(context)
    spec_with_name = tensor_spec.TensorSpec(
        [11, 3, 5], dtype=dtypes.int32,
        name='name').__tf_tracing_type__(context)

    self.assertEqual(tensor, spec)
    self.assertNotEqual(tensor, spec_with_name)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTensorShapeUnknown(self):
    # [FUNCTION DOC]
    # Scenario: two specs with completely unknown shape. Expects equal TraceTypes.
    context = trace_type.InternalTracingContext()
    spec_1 = tensor_spec.TensorSpec(
        None, dtype=dtypes.int32).__tf_tracing_type__(context)
    spec_2 = tensor_spec.TensorSpec(
        None, dtype=dtypes.int32).__tf_tracing_type__(context)
    self.assertEqual(spec_1, spec_2)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testAttrsTraceTypeGeneration(self):
    # [FUNCTION DOC]
    # Scenario: an attrs-like instance with two int attributes. Expects the generated type to
    # equal an Attrs type built by hand from Literals and to be a subtype of itself.
    trace_a = trace_type.from_value(TestAttrsClass(1, 2))
    expected = default_types.Attrs.from_type_and_attributes(
        TestAttrsClass, (default_types.Literal(1), default_types.Literal(2)))
    self.assertEqual(trace_a, expected)
    self.assertTrue(trace_a.is_subtype_of(trace_a))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTupleEquality(self):
    # [FUNCTION DOC]
    # Scenario: tuples with different elements or lengths, and an identical one. Expects types to
    # differ unless the tuples are identical.
    trace_a = trace_type.from_value((1, 2, 3, 4))
    trace_b = trace_type.from_value((1, 2, 2, 4))
    trace_c = trace_type.from_value((1, 2, 3))
    trace_d = trace_type.from_value((1, 2, 3, 4))
    self.assertNotEqual(trace_a, trace_b)
    self.assertNotEqual(trace_a, trace_c)
    self.assertNotEqual(trace_b, trace_c)
    self.assertEqual(trace_a, trace_d)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testListEquality(self):
    # [FUNCTION DOC]
    # Scenario: lists with different elements or lengths, and an identical one. Expects types to
    # differ unless the lists are identical.
    trace_a = trace_type.from_value([1, 2, 3, 4])
    trace_b = trace_type.from_value([1, 2, 2, 4])
    trace_c = trace_type.from_value([1, 2, 3])
    trace_d = trace_type.from_value([1, 2, 3, 4])
    self.assertNotEqual(trace_a, trace_b)
    self.assertNotEqual(trace_a, trace_c)
    self.assertNotEqual(trace_b, trace_c)
    self.assertEqual(trace_a, trace_d)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testDictEquality(self):
    # [FUNCTION DOC]
    # Scenario: dictionaries with different values and one identical dictionary with a different
    # insertion order. Expects differing values to give unequal types and insertion order to be
    # irrelevant.
    trace_a = trace_type.from_value({1: 2, 3: 4})
    trace_b = trace_type.from_value({1: 2, 3: 2})
    trace_c = trace_type.from_value({1: 2, 3: 0})
    trace_d = trace_type.from_value({3: 4, 1: 2})
    self.assertNotEqual(trace_a, trace_b)
    self.assertNotEqual(trace_a, trace_c)
    self.assertNotEqual(trace_b, trace_c)
    self.assertEqual(trace_a, trace_d)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testComplexStruct(self):
    # [FUNCTION DOC]
    # Scenario: a deeply nested structure of dicts and tuples, converted twice. Expects equal
    # types that are subtypes of each other.
    struct = {(1, 2, 3): {(1, 2): {12: 2}}, (3, 2, 3): (2, {2: 3})}
    trace_a = trace_type.from_value(struct)
    trace_b = trace_type.from_value(struct)
    self.assertEqual(trace_a, trace_b)
    self.assertTrue(trace_a.is_subtype_of(trace_b))
    self.assertTrue(trace_b.is_subtype_of(trace_a))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testCustomUnequableTypeSucceeds(self):
    # [FUNCTION DOC]
    # Scenario: objects whose equality always raises an error. Expects TraceTypes of the same live
    # object to be equal, comparing types of two different live objects to propagate the error,
    # and types to compare unequal once the referenced object has been deleted.

    class CustomUnequable:
      # [CLASS DOC]
      # HELPER: Local fixture whose equality check always raises and whose hash is constant.

      def __eq__(self, o):
        # [FUNCTION DOC]
        # Always raises a value error.
        raise ValueError

      def __hash__(self):
        # [FUNCTION DOC]
        # Returns the constant zero.
        return 0

    object_a = CustomUnequable()
    object_b = CustomUnequable()
    trace_a_1 = trace_type.from_value(object_a)
    trace_a_2 = trace_type.from_value(object_a)
    trace_b = trace_type.from_value(object_b)
    self.assertEqual(trace_a_1, trace_a_2)

    with self.assertRaises(ValueError):
      trace_a_1.__eq__(trace_b)

    del object_a
    self.assertNotEqual(trace_a_1, trace_a_2)
    self.assertNotEqual(trace_a_2, trace_a_1)

    del object_b
    self.assertNotEqual(trace_a_1, trace_a_2)
    self.assertNotEqual(trace_a_2, trace_a_1)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testCustomUnhashableTypeFailsGracefully(self):
    # [FUNCTION DOC]
    # Scenario: an object defining equality but no hash (so unhashable). Expects building its
    # TraceType to raise a type error saying a generic TraceType could not be generated.

    class CustomUnhashable:
      # [CLASS DOC]
      # HELPER: Local fixture defining equality only, which makes instances unhashable.

      def __eq__(self, o):
        # [FUNCTION DOC]
        # Always reports equality.
        return True

    obj = CustomUnhashable()
    with self.assertRaisesRegex(
        TypeError,
        r'Could not generate a generic TraceType for'):
      trace_type.from_value(obj)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testGetDefaultPlaceholderValue(self):
    # [FUNCTION DOC]
    # Scenario: a nested value mixing lists, tuples, dicts and an attrs-like instance. Builds its
    # TraceType and a placeholder with a default placeholder context. Expects the placeholder to
    # equal the original value.
    placeholder_context = trace_type.InternalPlaceholderContext()
    composite_value = [1, 2, (3, [4, 5]), {6: [7]}, TestAttrsClass(8, (10, 11))]
    composite_type = trace_type.from_value(composite_value)
    placeholder_value = composite_type.placeholder_value(placeholder_context)
    self.assertEqual(composite_value, placeholder_value)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testWrappedNamedTuple(self):
    # [FUNCTION DOC]
    # Scenario: an empty tuple subclass that exposes a named tuple through the wrapped attribute
    # (like trackable data structures). Expects its TraceType to equal that of the underlying
    # named tuple.
    ActualType = collections.namedtuple('ActualType', ['a', 'b', 'c'])

    class MockWrapper(tuple):
      # [CLASS DOC]
      # HELPER: Local fixture tuple subclass that wraps a named tuple instance via the wrapped
      # attribute.
      # Generated through trackable data structures:
      # //tensorflow/python/trackable/data_structures.py
      # With design pattern similar to Python functools:
      # https://docs.python.org/3/library/functools.html?highlight=__wrapped__#functools.update_wrapper
      __wrapped__ = ActualType(1, 2, 3)

    self.assertEqual(
        trace_type.from_value(MockWrapper()),
        trace_type.from_value(ActualType(1, 2, 3)))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testBadReturnType(self):
    # [FUNCTION DOC]
    # Scenario: an object whose tracing-protocol method returns a non-TraceType value. Expects
    # building its TraceType to raise a type error.
    class MyClass:
      # [CLASS DOC]
      # HELPER: Local fixture implementing the tracing protocol incorrectly.

      def __tf_tracing_type__(self, _):
        # [FUNCTION DOC]
        # Ignores the context and returns the integer one, which is not a TraceType.
        return 1

    with self.assertRaises(TypeError):
      trace_type.from_value(MyClass())


class CastDefaultTypesTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the cast operation of TraceTypes produced by the builder: values of tensors,
  # containers, attrs, numpy arrays, custom nest types and weak references are cast against a
  # default cast context.

  def testLiteral(self):
    # [FUNCTION DOC]
    # Scenario: a Literal of 1.5. Casting 1.5 is expected to return it; casting integer 1 is
    # expected to raise a value error.
    trace_float = default_types.Literal(1.5)
    ctx = trace_type.InternalCastContext()
    value = trace_float.cast(1.5, ctx)
    self.assertEqual(value, 1.5)
    with self.assertRaises(ValueError):
      _ = trace_float.cast(1, ctx)

  @parameterized.parameters(list, tuple)
  def testTupleAndList(self, container_type):
    # [FUNCTION DOC]
    # Parameterized over list and tuple. Scenario: a container of two float32 tensors is turned
    # into a TraceType, then a container of two Python ints is cast to it. Expects the result to
    # be the same container kind with the same length and values, with elements converted to
    # float32 tensors.
    foo = (
        constant_op.constant(1.0, dtypes.float32),
        constant_op.constant(2.0, dtypes.float32))
    foo = container_type(foo)
    trace_foo = trace_type.from_value(foo)
    bar = (1, 2)
    bar = container_type(bar)
    ctx = trace_type.InternalCastContext()
    value = trace_foo.cast(bar, ctx)

    self.assertIsInstance(value, container_type)
    self.assertLen(value, len(bar))
    self.assertSequenceEqual(value, bar)
    self.assertEqual(value[0].dtype, dtypes.float32)
    self.assertEqual(value[1].dtype, dtypes.float32)

  @parameterized.parameters(
      (list, tuple),
      (tuple, list))
  def testTupleAndListCannotBeCasted(self, type_a, type_b):
    # [FUNCTION DOC]
    # Parameterized over (list, tuple) and (tuple, list). Scenario: a TraceType from a container
    # of one kind is asked to cast a container of the other kind. Expects an assertion error.
    foo = (
        constant_op.constant(1.0, dtypes.float32),
        constant_op.constant(2.0, dtypes.float32))
    foo = type_a(foo)
    trace_foo = trace_type.from_value(foo)
    bar = (1, 2)
    bar = type_b(bar)
    ctx = trace_type.InternalCastContext()
    with self.assertRaises(AssertionError):
      _ = trace_foo.cast(bar, ctx)

  def testNamedTuple(self):
    # [FUNCTION DOC]
    # Scenario: a named tuple of float32 tensors defines the TraceType; a named tuple of ints is
    # cast. Expects the same named tuple class, same length and values, and float32 elements.
    Foo = collections.namedtuple('Foo', ['x', 'y'])
    foo = Foo(
        constant_op.constant(1.0, dtypes.float32),
        constant_op.constant(2.0, dtypes.float32))
    trace_foo = trace_type.from_value(foo)
    bar = Foo(1, 2)
    ctx = trace_type.InternalCastContext()
    value = trace_foo.cast(bar, ctx)

    self.assertIsInstance(value, Foo)
    self.assertLen(value, len(bar))
    self.assertSequenceEqual(value, bar)
    self.assertEqual(value[0].dtype, dtypes.float32)
    self.assertEqual(value[1].dtype, dtypes.float32)

  def testAttrs(self):
    # [FUNCTION DOC]
    # Scenario: an attrs-like instance of float32 tensors defines the TraceType; an instance of
    # ints is cast. Expects an instance of the same class whose attributes are float32 tensors.
    foo = TestAttrsClass(
        constant_op.constant(1.0, dtypes.float32),
        constant_op.constant(2.0, dtypes.float32),)
    trace_foo = trace_type.from_value(foo)
    bar = TestAttrsClass(1, 2)
    ctx = trace_type.InternalCastContext()
    value = trace_foo.cast(bar, ctx)

    self.assertIsInstance(value, TestAttrsClass)
    self.assertEqual(value.a.dtype, dtypes.float32)
    self.assertEqual(value.b.dtype, dtypes.float32)

  def testDict(self):
    # [FUNCTION DOC]
    # Scenario: a dict of float32 tensors defines the TraceType; a dict of ints with the same
    # keys is cast. Expects a dict with the same keys whose values are float32 tensors.
    foo = {'x': constant_op.constant(1.0, dtypes.float32),
           'y': constant_op.constant(2.0, dtypes.float32)}
    trace_foo = trace_type.from_value(foo)
    bar = {'x': 1, 'y': 2}
    ctx = trace_type.InternalCastContext()
    value = trace_foo.cast(bar, ctx)

    self.assertIsInstance(value, dict)
    self.assertSequenceEqual(
        set(value.keys()),
        set(bar.keys()))
    self.assertIn('x', value)
    self.assertIn('y', value)
    self.assertEqual(value['x'].dtype, dtypes.float32)
    self.assertEqual(value['y'].dtype, dtypes.float32)

  def testNumpy(self):
    # [FUNCTION DOC]
    # Scenario: a numpy array. Expects its TraceType to equal the registered tensor type built
    # from the array's shape and dtype.
    ndarray = np.array([1, 2, 3])
    ndarray_type = trace_type.from_value(ndarray)
    self.assertEqual(
        ndarray_type, default_types.TENSOR(ndarray.shape, ndarray.dtype)
    )

  def testWeakrefInput(self):
    # [FUNCTION DOC]
    # Scenario: a weak reference object passed as the value. Expects a type error saying weak
    # reference input is not supported.
    obj = DummyGenericClass()
    ref = weakref.ref(obj)
    with self.assertRaisesRegex(TypeError, 'weakref input .* not supported'):
      trace_type.from_value(ref)

  def testCustomNestCast(self):
    # [FUNCTION DOC]
    # Scenario: a custom nest value with a float32 tensor defines the TraceType (expected to be
    # the custom nest TraceType); another with the same mask but a plain list is cast. Expects an
    # instance of the same class with an equal mask and a value converted to a tensor of the
    # original dtype and shape with the new contents.
    mt = MaskedTensor(
        mask=False, value=constant_op.constant([1.0], dtype=dtypes.float32)
    )
    mt2 = MaskedTensor(mask=False, value=[2])
    ctx = trace_type.InternalCastContext()

    trace_mt = trace_type.from_value(mt)
    self.assertIsInstance(trace_mt, custom_nest_trace_type.CustomNestTraceType)

    mt2_casted = trace_mt.cast(mt2, ctx)
    self.assertIsInstance(mt2_casted, MaskedTensor)
    self.assertEqual(mt2_casted.mask, mt2.mask)
    self.assertEqual(mt2_casted.value.dtype, mt.value.dtype)
    self.assertAllEqual(mt2_casted.value.shape, mt.value.shape)
    self.assertAllEqual(mt2_casted.value, mt2.value)

  def testCustomNestFailCastWithWrongMetadata(self):
    # [FUNCTION DOC]
    # Scenario: casting a custom nest value whose mask differs from the one in the TraceType.
    # Expects a value error mentioning a metadata mismatch.
    mt = MaskedTensor(mask=False, value=constant_op.constant([1.0]))
    mt2 = MaskedTensor(mask=True, value=constant_op.constant([1.0]))
    ctx = trace_type.InternalCastContext()
    trace_mt = trace_type.from_value(mt)
    with self.assertRaisesRegex(ValueError, 'Metadata mismatch'):
      trace_mt.cast(mt2, ctx)


class SignatureToTraceTypeTest(test.TestCase):
  # [CLASS DOC]
  # Tests that, with a legacy-signature tracing context, existing specs are passed through as
  # TraceTypes (including inside lists and dicts).

  def testTensorSpecs(self):
    # [FUNCTION DOC]
    # Scenario: an unknown-shape tensor spec under a legacy-signature context. Expects the result
    # to equal the spec itself.
    self.assertEqual(
        trace_type.from_value(
            tensor_spec.TensorSpec(shape=None),
            trace_type.InternalTracingContext(is_legacy_signature=True)),
        tensor_spec.TensorSpec(shape=None))

  def testListofTensorSpecs(self):
    # [FUNCTION DOC]
    # Scenario: a list of two specs under a legacy-signature context. Expects a List TraceType of
    # those two specs.
    self.assertEqual(
        trace_type.from_value([
            tensor_spec.TensorSpec(shape=None),
            tensor_spec.TensorSpec(shape=None)
        ], trace_type.InternalTracingContext(is_legacy_signature=True)),
        default_types.List(
            tensor_spec.TensorSpec(shape=None),
            tensor_spec.TensorSpec(shape=None)))

  def testDictofTensorSpecs(self):
    # [FUNCTION DOC]
    # Scenario: a dict of two specs under a legacy-signature context. Expects a Dict TraceType
    # with the same keys and those specs.
    self.assertEqual(
        trace_type.from_value(
            {
                'a': tensor_spec.TensorSpec(shape=None),
                'b': tensor_spec.TensorSpec(shape=None)
            }, trace_type.InternalTracingContext(is_legacy_signature=True)),
        default_types.Dict({
            'a': tensor_spec.TensorSpec(shape=None),
            'b': tensor_spec.TensorSpec(shape=None)
        }))


class TraceTypeMemoryTest(test.TestCase):
  # [CLASS DOC]
  # Memory-leak tests: each test builds TraceTypes for a kind of value while a decorator checks
  # that no new Python objects remain alive after eager execution.

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testGeneric(self):
    # [FUNCTION DOC]
    # Builds TraceTypes for an int and for a generic object; asserts nothing itself (the
    # decorator checks for leaked objects).
    trace_type.from_value(1)
    trace_type.from_value(DummyGenericClass())

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testTensor(self):
    # [FUNCTION DOC]
    # Builds the TraceType of a zeros tensor; leak check by decorator only.
    tensor = array_ops.zeros([10])
    trace_type.from_value(tensor)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testTuple(self):
    # [FUNCTION DOC]
    # Builds the TraceType of a tuple of ints; leak check by decorator only.
    trace_type.from_value((1, 2, 3))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDict(self):
    # [FUNCTION DOC]
    # Builds the TraceType of a dict of ints; leak check by decorator only.
    trace_type.from_value({1: 1, 2: 2, 3: 3})

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testList(self):
    # [FUNCTION DOC]
    # Builds the TraceType of a list of ints; leak check by decorator only.
    trace_type.from_value([1, 2, 3])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testAttrs(self):
    # [FUNCTION DOC]
    # Builds the TraceType of an attrs-like instance; leak check by decorator only.
    trace_type.from_value(TestAttrsClass(1, 2))


class TraceTypeGenerationBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmarks measuring the average time to generate TraceTypes (cache keys) for tensors,
  # specs, variables, nested structures and for repeated invocation of a traced function. Each
  # benchmark times many iterations and reports the wall time and an average in milliseconds.

  def benchmarkTensor(self):
    # [FUNCTION DOC]
    # Times 100000 generations of the TraceType of a list of four zeros tensors of different
    # shapes and reports the benchmark with the average time per generation in milliseconds.
    shapes = [[1], [2, 19], [5, 11, 24], [4, 5, 9, 23]]
    tensors = []
    for s in shapes:
      tensors.append(array_ops.zeros(s))

    def encode_tensors(tensors):
      # [FUNCTION DOC]
      # HELPER: Timed body for the tensor benchmark; builds the TraceType of the given tensors.
      trace_type.from_value(tensors)

    iterations = 100000
    t = timeit.timeit(lambda: encode_tensors(tensors), number=iterations)
    self.report_benchmark(
        name='tensor_cache_key_generation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'tensor_cache_key_generation_avg_ms',
            'value': t / iterations * 1000
        }])

  def benchmarkTensorSpec(self):
    # [FUNCTION DOC]
    # Times 100000 generations of the TraceType of a list of four int32 tensor specs and reports
    # the average time per generation in milliseconds.
    shapes = [[1], [2, 19], [5, 11, 24], [4, 5, 9, 23]]
    tensor_specs = []
    for s in shapes:
      tensor_specs.append(tensor_spec.TensorSpec(s, dtypes.int32))

    def encode_tensor_specs(tensor_specs):
      # [FUNCTION DOC]
      # HELPER: Timed body for the spec benchmark; builds the TraceType of the given specs.
      trace_type.from_value(tensor_specs)

    iterations = 100000
    t = timeit.timeit(
        lambda: encode_tensor_specs(tensor_specs), number=iterations)
    self.report_benchmark(
        name='tensor_spec_cache_key_generation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'tensor_spec_cache_key_generation_avg_ms',
            'value': t / iterations * 1000
        }])

  def benchmarkVariable(self):
    # [FUNCTION DOC]
    # Times 10000 generations of the TraceType of a list of three variables and reports the
    # average time per generation in milliseconds.
    var_list = [
        variables.Variable(1.0),
        variables.Variable(1),
        variables.Variable([1])
    ]

    def encode_variables(var_list):
      # [FUNCTION DOC]
      # HELPER: Timed body for the variable benchmark; builds the TraceType of the given list.
      trace_type.from_value(var_list)

    iterations = 10000
    t = timeit.timeit(lambda: encode_variables(var_list), number=iterations)
    self.report_benchmark(
        name='variable_cache_key_generation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'variable_cache_key_generation_avg_ms',
            'value': t / iterations * 1000
        }])

  def benchmarkTraceTypeLookup(self):
    # [FUNCTION DOC]
    # Defines a traced function, calls it with several argument shapes to fill its cache, then
    # times 10000 calls with an argument that hits the cache and reports the average lookup time
    # in milliseconds.

    @def_function.function
    def defined(t):
      # [FUNCTION DOC]
      # HELPER: Identity function traced for the lookup benchmark; returns its argument.
      return t

    call_arg_list = [
        1,
        array_ops.zeros([5, 13]),
        array_ops.zeros([9, 22, 24]),
        array_ops.zeros([5, 13, 2])
    ]

    for c in call_arg_list:
      defined(c)

    lookup_call_arg = array_ops.zeros([5, 13])

    iterations = 10000
    t = timeit.timeit(stmt=lambda: defined(lookup_call_arg), number=iterations)

    self.report_benchmark(
        name='cache_key_lookup',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'cache_key_lookup_avg_ms',
            'value': t / iterations * 1000
        }])

  def benchmarkNestedStruct(self):
    # [FUNCTION DOC]
    # Times 100000 generations of the TraceType of a deeply nested dict and tuple structure and
    # reports the average time per generation in milliseconds.
    struct = {(1, 2, 3): {(1, 2): {12: 2}}, (3, 2, 3): (2, {2: 3})}

    def encode_struct(struct):
      # [FUNCTION DOC]
      # HELPER: Timed body for the nested structure benchmark; builds the TraceType of the struct.
      trace_type.from_value(struct)

    iterations = 100000
    t = timeit.timeit(lambda: encode_struct(struct), number=iterations)
    self.report_benchmark(
        name='nested_struct_cache_key_generation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'nested_struct_cache_key_generation_avg_ms',
            'value': t / iterations * 1000
        }])

  def benchmarkFunctionInvocation(self):
    # [FUNCTION DOC]
    # Builds a structure of variables and tensors, calls a traced identity function once to
    # trace and cache it, then times 10000 further calls and reports the average invocation time
    # in milliseconds.
    struct = (variables.Variable(1.0), array_ops.zeros([5, 13]), {
        'tensor': array_ops.zeros([5, 20]),
        'variable': variables.Variable(1.0)
    })

    @def_function.function
    def defined(t):
      # [FUNCTION DOC]
      # HELPER: Identity function traced for the invocation benchmark; returns its argument.
      return t

    defined(struct)  # Get it traced and cached.

    iterations = 10000
    t = timeit.timeit(lambda: defined(struct), number=iterations)
    self.report_benchmark(
        name='function_invocation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'function_invocation_time_avg_ms',
            'value': t / iterations * 1000
        }])


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
