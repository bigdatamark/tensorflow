# [MODULE DOC]
# Tests for CustomNestTraceType. Defines mock component trace types and a small data class
# implementing the custom nest protocol (flatten/unflatten with a mask as metadata and two
# values as components), then checks equality, placeholder creation, subtype and supertype
# relations.
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
"""Tests for CustomNestTraceType."""

import dataclasses
from typing import Any

from tensorflow.core.function.trace_type import custom_nest_trace_type
from tensorflow.core.function.trace_type import default_types
from tensorflow.python.platform import test
from tensorflow.python.types import trace


class MockSupertypes2With3(trace.TraceType):
  # [CLASS DOC]
  # HELPER: mock component trace type for the tests.
  # Wraps a value; the value 2 is a subtype of 3 only, and the common supertype of a type
  # holding 2 with an integer-holding type is a type holding 3 (otherwise nothing).

  def __init__(self, obj):
    # [FUNCTION DOC]
    # Stores the wrapped object.
    self._object = obj

  def is_subtype_of(self, other):
    # [FUNCTION DOC]
    # True only when this wraps 2 and the other wraps 3.
    return self._object == 2 and other._object == 3

  def most_specific_common_supertype(self, others):
    # [FUNCTION DOC]
    # Returns itself for an empty sequence. If this wraps 2 and the first other wraps an
    # integer, returns a new mock wrapping 3; otherwise returns nothing. Only the first
    # element of others is inspected.
    if not others:
      return self

    if self._object == 2 and isinstance(others[0]._object, int):
      return MockSupertypes2With3(3)
    else:
      return None

  def placeholder_value(self, placeholder_context=None):
    # [FUNCTION DOC]
    # Not supported by this mock; always raises a not-implemented error.
    raise NotImplementedError

  def __eq__(self, other) -> bool:
    # [FUNCTION DOC]
    # Equal when the other is an instance of this mock's type and wraps an equal object.
    return isinstance(other, type(self)) and self._object == other._object

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # KNOWN ISSUE: reads an attribute named object-hash that is never defined, so calling it
    # raises an attribute error; the tests evidently never hash these mocks (the nest trace
    # type hash is not exercised here).
    # Intended to return a hash of the wrapped object.
    return self._object_hash


class Mock2AsTopType(MockSupertypes2With3):
  # [CLASS DOC]
  # HELPER: mock component trace type for the tests, where the value 2 acts as the top of the
  # hierarchy: anything is a subtype of a type wrapping 2.

  def is_subtype_of(self, other):
    # [FUNCTION DOC]
    # True when the other wraps 2.
    return other._object == 2

  def most_specific_common_supertype(self, others):
    # [FUNCTION DOC]
    # Returns nothing if any other is not this mock type. Otherwise returns itself if all
    # others wrap the same object, else a new mock wrapping 2 (the top).
    if not all(isinstance(other, Mock2AsTopType) for other in others):
      return None
    return (
        self
        if all(self._object == other._object for other in others)
        else Mock2AsTopType(2)
    )


@dataclasses.dataclass
class MaskedValuePair:
  # [CLASS DOC]
  # HELPER: sample data class implementing the custom nest protocol for the tests. Fields: a
  # boolean mask (used as metadata) and two values (the components).
  mask: bool
  value1: Any
  value2: Any

  def __tf_flatten__(self):
    # [FUNCTION DOC]
    # Returns the metadata (a one-element tuple holding the mask) and the components (a tuple
    # of the two values).
    metadata = (self.mask,)
    components = (self.value1, self.value2)
    return metadata, components

  @classmethod
  def __tf_unflatten__(cls, metadata, leaves):
    # [FUNCTION DOC]
    # Class-level constructor inverse of flatten: builds a new instance with the mask from the
    # metadata's first element and the two values from the first two leaves.
    mask = metadata[0]
    value1 = leaves[0]
    value2 = leaves[1]
    return MaskedValuePair(mask=mask, value1=value1, value2=value2)


class CustomNestTraceTypeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for CustomNestTraceType using the sample data class and mock component types.

  def testCustomNestTraceTypeEq(self):
    # [FUNCTION DOC]
    # Asserts two types with equal class, metadata and literal components are equal, and that a
    # type with different metadata and components is not.
    trace_components1 = (default_types.Literal(1.0), default_types.Literal(2.0))
    t1 = custom_nest_trace_type.CustomNestTraceType(
        MaskedValuePair, (True,), trace_components1
    )

    trace_components2 = (default_types.Literal(1.0), default_types.Literal(2.0))
    t2 = custom_nest_trace_type.CustomNestTraceType(
        MaskedValuePair, (True,), trace_components2
    )
    self.assertEqual(t1, t2)

    trace_components3 = (default_types.Literal(2), default_types.Literal(1))
    t3 = custom_nest_trace_type.CustomNestTraceType(
        MaskedValuePair, (False,), trace_components3
    )
    self.assertNotEqual(t1, t3)

  def testCustomNestTraceTypePlaceholderValue(self):
    # [FUNCTION DOC]
    # Creates placeholder values (no context) from two types with literal components and
    # asserts the rebuilt data class instances have the expected mask and values.
    trace_components1 = (default_types.Literal(1.0), default_types.Literal(2.0))
    t1 = custom_nest_trace_type.CustomNestTraceType(
        MaskedValuePair, (True,), trace_components1
    )

    mvp1 = MaskedValuePair(mask=True, value1=1.0, value2=2.0)
    phv1 = t1.placeholder_value(None)
    self.assertEqual(phv1.mask, mvp1.mask)
    self.assertEqual(phv1.value1, mvp1.value1)
    self.assertEqual(phv1.value2, mvp1.value2)

    trace_components2 = (default_types.Literal(2), default_types.Literal(1))
    t2 = custom_nest_trace_type.CustomNestTraceType(
        MaskedValuePair, (False,), trace_components2
    )
    mvp2 = MaskedValuePair(mask=False, value1=2, value2=1)
    phv2 = t2.placeholder_value(None)
    self.assertEqual(phv2.mask, mvp2.mask)
    self.assertEqual(phv2.value1, mvp2.value1)
    self.assertEqual(phv2.value2, mvp2.value2)

  def testCustomNestTraceTypeSubtype(self):
    # [FUNCTION DOC]
    # Uses top-type mock components to assert subtype relations: componentwise lower types are
    # subtypes of top-valued ones but not vice versa, equal types are mutual subtypes, and a
    # mixed component set is unrelated in both directions.
    trace_components1 = (Mock2AsTopType(1), Mock2AsTopType(1))
    t1 = custom_nest_trace_type.CustomNestTraceType(
        MaskedValuePair, (True,), trace_components1
    )

    trace_components2 = (Mock2AsTopType(2), Mock2AsTopType(2))
    t2 = custom_nest_trace_type.CustomNestTraceType(
        MaskedValuePair, (True,), trace_components2
    )
    self.assertTrue(t1.is_subtype_of(t2))
    self.assertFalse(t2.is_subtype_of(t1))

    trace_components3 = (Mock2AsTopType(2), Mock2AsTopType(2))
    t4 = custom_nest_trace_type.CustomNestTraceType(
        MaskedValuePair, (True,), trace_components3
    )
    self.assertTrue(t2.is_subtype_of(t4))
    self.assertTrue(t4.is_subtype_of(t2))

    trace_components4 = (Mock2AsTopType(1), Mock2AsTopType(2))
    t4 = custom_nest_trace_type.CustomNestTraceType(
        MaskedValuePair, (True,), trace_components4
    )
    self.assertFalse(t1.is_subtype_of(t4))
    self.assertFalse(t4.is_subtype_of(t1))

  def testCustomNestTraceTypeSupertype(self):
    # [FUNCTION DOC]
    # Asserts the common supertype of types with components 2 and 1 is the type with
    # components 3, and that when components lack a common supertype the result has null
    # components.
    trace_components1 = (MockSupertypes2With3(2), MockSupertypes2With3(2))
    t1 = custom_nest_trace_type.CustomNestTraceType(
        MaskedValuePair, (True,), trace_components1
    )

    trace_components2 = (MockSupertypes2With3(1), MockSupertypes2With3(1))
    t2 = custom_nest_trace_type.CustomNestTraceType(
        MaskedValuePair, (True,), trace_components2
    )

    trace_components3 = (MockSupertypes2With3(3), MockSupertypes2With3(3))
    t3 = custom_nest_trace_type.CustomNestTraceType(
        MaskedValuePair, (True,), trace_components3
    )
    self.assertEqual(t1.most_specific_common_supertype([t2]), t3)

    c_super_none = custom_nest_trace_type.CustomNestTraceType(
        MaskedValuePair, (True,), (None, None)
    )
    self.assertEqual(t2.most_specific_common_supertype([t3]), c_super_none)
    self.assertEqual(t3.most_specific_common_supertype([t1]), c_super_none)


if __name__ == '__main__':
  test.main()
