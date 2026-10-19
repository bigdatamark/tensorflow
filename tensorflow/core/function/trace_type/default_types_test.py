# [MODULE DOC]
# Unit tests for the default TraceType implementations (Literal, List, Tuple, NamedTuple, Attrs
# and Dict) in the default types module. Covers NaN handling, most-specific-common-supertype
# results, subtype checks, string forms, serialization round trips, list versus tuple
# inequality, dictionary equality and lazy casting. Defines small mock TraceTypes and an
# attrs-like helper class used as fixtures. Runs through the TensorFlow test runner.
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
"""Tests for default_types."""

import collections

from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.platform import test
from tensorflow.python.types import trace


class MockSupertypes2With3(trace.TraceType):
  # [CLASS DOC]
  # HELPER: Mock TraceType fixture for the default types tests.
  # A fake TraceType wrapping an arbitrary object. Object 2 is a subtype of object 3, and the
  # common supertype of 2 and any integer is 3. Used to exercise the container TraceTypes.

  def __init__(self, obj):
    # [FUNCTION DOC]
    # Stores the wrapped object (any value).
    self._object = obj

  def is_subtype_of(self, other):
    # [FUNCTION DOC]
    # True only if this wraps 2 and the other wraps 3.
    return self._object == 2 and other._object == 3

  def most_specific_common_supertype(self, others):
    # [FUNCTION DOC]
    # Returns itself when no others are given; if this wraps 2 and the first other wraps an
    # integer, returns a new mock wrapping 3; otherwise returns nothing.
    if not others:
      return self

    if self._object == 2 and isinstance(others[0]._object, int):
      return MockSupertypes2With3(3)
    else:
      return None

  def placeholder_value(self, placeholder_context=None):
    # [FUNCTION DOC]
    # Not supported by the mock: always raises a not-implemented error.
    raise NotImplementedError

  def __eq__(self, other) -> bool:
    # [FUNCTION DOC]
    # True if the other is an instance of the same type wrapping an equal object.
    return isinstance(other, type(self)) and self._object == other._object

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # KNOWN ISSUE: reads an attribute that is never set (the object hash), so calling it would
    # raise an attribute error. It works in the tests only because they never hash the mocks.
    return self._object_hash

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # Returns the fixed name of the mock class.
    return 'MockSupertypes2With3'


class Mock2AsTopType(MockSupertypes2With3):
  # [CLASS DOC]
  # HELPER: Mock TraceType fixture for the default types tests.
  # Variant of the mock where the wrapped object 2 acts as the top of the type order: anything
  # is a subtype of a mock wrapping 2, and the common supertype of differing values is 2.

  def is_subtype_of(self, other):
    # [FUNCTION DOC]
    # True if the other mock wraps 2.
    return other._object == 2

  def most_specific_common_supertype(self, others):
    # [FUNCTION DOC]
    # Returns nothing if any other is not this mock type; returns itself if all wrap the same
    # object; otherwise a new mock wrapping 2.
    if not all(isinstance(other, Mock2AsTopType) for other in others):
      return None
    return (
        self
        if all(self._object == other._object for other in others)
        else Mock2AsTopType(2)
    )

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # Returns the fixed name of this mock class.
    return 'Mock2AsTopType'


class TestAttr:
  # [CLASS DOC]
  # HELPER: Fixture imitating a single attrs attribute descriptor; only carries a name.
  """Helps test attrs collections."""

  def __init__(self, name):
    # [FUNCTION DOC]
    # Stores the attribute name.
    self.name = name


class TestAttrsClass:
  # [CLASS DOC]
  # HELPER: Fixture imitating a class decorated with the attrs library: it exposes the attrs
  # metadata list naming attributes a and b and stores two values. Used to test the Attrs type.
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
    return (
        isinstance(other, TestAttrsClass)
        and self.a == other.a
        and self.b == other.b
    )


class DefaultTypesTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the default TraceType classes (Literal, List, Tuple, NamedTuple, Attrs,
  # Dict). Uses the module-level mock TraceTypes as fixtures; no per-test setup.

  def testLiteralNan(self):
    # [FUNCTION DOC]
    # Scenario: Literals built from a float NaN and from complex numbers with a NaN real or
    # imaginary part. Expects all of them to compare equal to each other.
    nan_literal = default_types.Literal(float('nan'))
    complex_nan = default_types.Literal(complex(float('nan'), 1))
    complex_nan_other = default_types.Literal(complex(1, float('nan')))
    self.assertEqual(nan_literal, nan_literal)
    self.assertEqual(nan_literal, complex_nan)
    self.assertEqual(nan_literal, complex_nan_other)

  def testLiteralSupertypes(self):
    # [FUNCTION DOC]
    # Scenario: Literals with equal and different values. Expects the common supertype with no
    # others or with equal literals to be the literal itself, and nothing with a different one.
    literal_a = default_types.Literal(1)
    literal_b = default_types.Literal(2)
    literal_c = default_types.Literal(1)

    self.assertEqual(literal_a, literal_a.most_specific_common_supertype([]))
    self.assertEqual(
        literal_a, literal_a.most_specific_common_supertype([literal_a])
    )
    self.assertEqual(
        literal_a, literal_a.most_specific_common_supertype([literal_c])
    )
    self.assertIsNone(literal_a.most_specific_common_supertype([literal_b]))

  def testLiteralSerialization(self):
    # [FUNCTION DOC]
    # Scenario: Literals of bool, int, float, string and None values. Expects the string form of
    # the bool literal to show its value and each literal to equal itself after a serialize and
    # deserialize round trip.
    literal_bool = default_types.Literal(True)
    literal_int = default_types.Literal(1)
    literal_float = default_types.Literal(1.2)
    literal_str = default_types.Literal('a')
    literal_none = default_types.Literal(None)

    self.assertEqual(str(literal_bool), 'Literal[True]')
    self.assertEqual(
        serialization.deserialize(serialization.serialize(literal_bool)),
        literal_bool,
    )
    self.assertEqual(
        serialization.deserialize(serialization.serialize(literal_int)),
        literal_int,
    )
    self.assertEqual(
        serialization.deserialize(serialization.serialize(literal_float)),
        literal_float,
    )
    self.assertEqual(
        serialization.deserialize(serialization.serialize(literal_str)),
        literal_str,
    )
    self.assertEqual(
        serialization.deserialize(serialization.serialize(literal_none)),
        literal_none,
    )

  def testListSupertype(self):
    # [FUNCTION DOC]
    # Scenario: two lists of mock types. Expects the supertype with no others to be the list
    # itself, nothing when a component has no supertype, and a list of the component-wise
    # supertypes otherwise.
    list_a = default_types.List(
        MockSupertypes2With3(1),
        MockSupertypes2With3(2),
        MockSupertypes2With3(3),
    )
    list_b = default_types.List(
        MockSupertypes2With3(2),
        MockSupertypes2With3(2),
        MockSupertypes2With3(2),
    )

    self.assertEqual(list_a, list_a.most_specific_common_supertype([]))
    self.assertIsNone(list_a.most_specific_common_supertype([list_b]))
    self.assertEqual(
        list_b.most_specific_common_supertype([list_a]),
        default_types.List(
            MockSupertypes2With3(3),
            MockSupertypes2With3(3),
            MockSupertypes2With3(3),
        ),
    )

  def testListSerialization(self):
    # [FUNCTION DOC]
    # Scenario: a list of three Literals. Expects a specific readable string and an equal list
    # after a serialization round trip.
    list_original = default_types.List(
        default_types.Literal(1),
        default_types.Literal(2),
        default_types.Literal(3),
    )

    self.assertEqual(
        str(list_original), 'List[Literal[1], Literal[2], Literal[3]]'
    )
    self.assertEqual(
        serialization.deserialize(serialization.serialize(list_original)),
        list_original,
    )

  def testTupleSupertype(self):
    # [FUNCTION DOC]
    # Scenario: two tuples of mock types. Expects the same results as for lists: itself with no
    # others, nothing when no supertype exists, and the component-wise supertype tuple otherwise.
    tuple_a = default_types.Tuple(
        MockSupertypes2With3(1),
        MockSupertypes2With3(2),
        MockSupertypes2With3(3),
    )
    tuple_b = default_types.Tuple(
        MockSupertypes2With3(2),
        MockSupertypes2With3(2),
        MockSupertypes2With3(2),
    )

    self.assertEqual(tuple_a, tuple_a.most_specific_common_supertype([]))
    self.assertIsNone(tuple_a.most_specific_common_supertype([tuple_b]))
    self.assertEqual(
        tuple_b.most_specific_common_supertype([tuple_a]),
        default_types.Tuple(
            MockSupertypes2With3(3),
            MockSupertypes2With3(3),
            MockSupertypes2With3(3),
        ),
    )

  def testTupleSerialization(self):
    # [FUNCTION DOC]
    # Scenario: a tuple of three Literals. Expects a specific readable string and an equal tuple
    # after a serialization round trip.
    tuple_original = default_types.Tuple(
        default_types.Literal(1),
        default_types.Literal(2),
        default_types.Literal(3),
    )

    self.assertEqual(
        str(tuple_original), 'Tuple[Literal[1], Literal[2], Literal[3]]'
    )
    self.assertEqual(
        serialization.deserialize(serialization.serialize(tuple_original)),
        tuple_original,
    )

  def testNamedTupleSupertype(self):
    # [FUNCTION DOC]
    # Scenario: two NamedTuple types built from the same named tuple class with mock attributes.
    # Expects a readable string with name and type pairs, the supertype with no others to be
    # itself, nothing when attributes lack a supertype, and the attribute-wise supertype
    # NamedTuple otherwise.
    named_tuple_type = collections.namedtuple('MyNamedTuple', 'x y z')
    tuple_a = default_types.NamedTuple.from_type_and_attributes(
        named_tuple_type,
        (
            MockSupertypes2With3(1),
            MockSupertypes2With3(2),
            MockSupertypes2With3(3),
        ),
    )
    tuple_b = default_types.NamedTuple.from_type_and_attributes(
        named_tuple_type,
        (
            MockSupertypes2With3(2),
            MockSupertypes2With3(2),
            MockSupertypes2With3(2),
        ),
    )

    self.assertEqual(
        str(tuple_a),
        "MyNamedTuple[['x', MockSupertypes2With3], ['y', MockSupertypes2With3],"
        " ['z', MockSupertypes2With3]]",
    )
    self.assertEqual(tuple_a, tuple_a.most_specific_common_supertype([]))
    self.assertIsNone(tuple_a.most_specific_common_supertype([tuple_b]))
    self.assertEqual(
        tuple_b.most_specific_common_supertype([tuple_a]),
        default_types.NamedTuple.from_type_and_attributes(
            named_tuple_type,
            (
                MockSupertypes2With3(3),
                MockSupertypes2With3(3),
                MockSupertypes2With3(3),
            ),
        ),
    )

  def testAttrsSupertype(self):
    # [FUNCTION DOC]
    # Scenario: two Attrs types built from the attrs-like fixture class with mock attributes.
    # Expects the same checks as for named tuples: readable string, itself with no others,
    # nothing when no supertype exists, and the attribute-wise supertype otherwise.
    attrs_a = default_types.Attrs.from_type_and_attributes(
        TestAttrsClass,
        (
            MockSupertypes2With3(1),
            MockSupertypes2With3(2),
            MockSupertypes2With3(3),
        ),
    )
    attrs_b = default_types.Attrs.from_type_and_attributes(
        TestAttrsClass,
        (
            MockSupertypes2With3(2),
            MockSupertypes2With3(2),
            MockSupertypes2With3(2),
        ),
    )

    self.assertEqual(
        str(attrs_a),
        "TestAttrsClass[['a', MockSupertypes2With3], ['b',"
        ' MockSupertypes2With3]]',
    )
    self.assertEqual(attrs_a, attrs_a.most_specific_common_supertype([]))
    self.assertIsNone(attrs_a.most_specific_common_supertype([attrs_b]))
    self.assertEqual(
        attrs_b.most_specific_common_supertype([attrs_a]),
        default_types.Attrs.from_type_and_attributes(
            TestAttrsClass,
            (
                MockSupertypes2With3(3),
                MockSupertypes2With3(3),
                MockSupertypes2With3(3),
            ),
        ),
    )

  def testDictTypeSubtype(self):
    # [FUNCTION DOC]
    # Scenario: dict types over the top-type mock, two with the same keys and one with fewer
    # keys. Expects the dict with smaller values to be a subtype of the other, and a dict with a
    # different key set to be a subtype of neither.
    dict_type = default_types.Dict
    dict_a = dict_type(
        {'a': Mock2AsTopType(1), 'b': Mock2AsTopType(1), 'c': Mock2AsTopType(1)}
    )
    dict_b = dict_type(
        {'a': Mock2AsTopType(2), 'b': Mock2AsTopType(2), 'c': Mock2AsTopType(2)}
    )
    dict_c = dict_type({'a': Mock2AsTopType(1), 'b': Mock2AsTopType(1)})

    self.assertTrue(dict_a.is_subtype_of(dict_b))
    self.assertFalse(dict_c.is_subtype_of(dict_b))
    self.assertFalse(dict_c.is_subtype_of(dict_a))

  def testDictTypeSupertype(self):
    # [FUNCTION DOC]
    # Scenario: two dict types with mock values. Expects itself with no others, nothing when a
    # key lacks a common supertype, and a dict of per-key supertypes otherwise.
    dict_type = default_types.Dict
    dict_a = dict_type({
        'a': MockSupertypes2With3(1),
        'b': MockSupertypes2With3(2),
        'c': MockSupertypes2With3(3),
    })
    dict_b = dict_type({
        'a': MockSupertypes2With3(2),
        'b': MockSupertypes2With3(2),
        'c': MockSupertypes2With3(2),
    })

    self.assertEqual(dict_a, dict_a.most_specific_common_supertype([]))
    self.assertIsNone(dict_a.most_specific_common_supertype([dict_b]))
    self.assertEqual(
        dict_b.most_specific_common_supertype([dict_a]),
        dict_type({
            'a': MockSupertypes2With3(3),
            'b': MockSupertypes2With3(3),
            'c': MockSupertypes2With3(3),
        }),
    )

  def testDictSerialization(self):
    # [FUNCTION DOC]
    # Scenario: a dict type with string keys and Literal values. Expects a specific readable
    # string and an equal dict type after a serialization round trip.
    dict_original = default_types.Dict({
        'a': default_types.Literal(1),
        'b': default_types.Literal(2),
        'c': default_types.Literal(3),
    })

    self.assertEqual(
        str(dict_original),
        "Dict[['a', Literal[1]], ['b', Literal[2]], ['c', Literal[3]]]",
    )
    self.assertEqual(
        serialization.deserialize(serialization.serialize(dict_original)),
        dict_original,
    )

  def testListTupleInequality(self):
    # [FUNCTION DOC]
    # Scenario: two equal lists and two equal tuples with the same Literal components. Expects
    # equal pairs to be equal and a list never to equal a tuple, in either direction.
    literal = default_types.Literal

    list_a = default_types.List(literal(1), literal(2), literal(3))
    list_b = default_types.List(literal(1), literal(2), literal(3))

    tuple_a = default_types.Tuple(literal(1), literal(2), literal(3))
    tuple_b = default_types.Tuple(literal(1), literal(2), literal(3))

    self.assertEqual(list_a, list_b)
    self.assertEqual(tuple_a, tuple_b)
    self.assertNotEqual(list_a, tuple_a)
    self.assertNotEqual(tuple_a, list_a)

  def testDictTypeEquality(self):
    # [FUNCTION DOC]
    # Scenario: dict types keyed by Literals, with the same entries in different insertion order
    # and one with fewer entries. Expects order to not matter and different entries to differ.
    dict_type = default_types.Dict
    literal = default_types.Literal

    dict_a = dict_type({literal(1): literal(2), literal(3): literal(4)})
    dict_b = dict_type({literal(1): literal(2)})
    dict_c = dict_type({literal(3): literal(4), literal(1): literal(2)})

    self.assertEqual(dict_a, dict_c)
    self.assertNotEqual(dict_a, dict_b)

  def testCastLazy(self):
    # [FUNCTION DOC]
    # Scenario: a dict type (placeholder type dict) nesting a tuple and a list of Literals.
    # A placeholder value is generated then cast back with the same type. Expects the cast to
    # return the very same object (no copy) because nothing needed changing.
    list_type = default_types.List(
        default_types.Literal('a'), default_types.Literal('b')
    )
    tuple_type = default_types.Tuple(default_types.Literal('c'), list_type)
    dict_type = default_types.Dict(
        {'key': tuple_type, 'other_key': list_type}, placeholder_type=dict
    )

    value = dict_type.placeholder_value(None)
    casted_value = dict_type.cast(value, None)

    self.assertIs(value, casted_value)


if __name__ == '__main__':
  test.main()
