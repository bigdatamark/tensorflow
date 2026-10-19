# [MODULE DOC]
# Unit tests for TypeDispatchTable (type_dispatch module). Defines a mock shape trace type and
# a helper to wrap it in a single-parameter function type, then tests target insertion order,
# duplicates, deletion, membership, exact and most-specific dispatch, cache invalidation on
# add/delete, insertion-order tie-breaking and the generalization heuristic.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for type_dispatch."""

from typing import Optional

from tensorflow.core.function.polymorphism import function_type
from tensorflow.core.function.polymorphism import type_dispatch
from tensorflow.python.platform import test
from tensorflow.python.types import trace


class MockShape(trace.TraceType):
  # [CLASS DOC]
  # HELPER: test double for shape trace types used by the dispatch tests.
  # Represents a shape as a tuple of dimensions where an unknown dimension (null) acts as a
  # wildcard. Subtype means same rank and each other-dimension is unknown or equal.

  def __init__(self, *shape: Optional[int]):
    # [FUNCTION DOC]
    # Stores the given dimensions (integers or null for unknown) as the shape attribute.
    self.shape = shape

  def is_subtype_of(self, other: "MockShape") -> bool:
    # [FUNCTION DOC]
    # Returns true if both shapes have the same rank and every dimension of the other shape is
    # unknown or equals the corresponding dimension of this one; false otherwise.
    if len(self.shape) != len(other.shape):
      return False

    return all(o is None or s == o for s, o in zip(self.shape, other.shape))

  def most_specific_common_supertype(self, others):
    # [FUNCTION DOC]
    # Returns nothing if any other shape has a different rank; otherwise returns a new shape
    # keeping each dimension where all shapes agree and using unknown elsewhere.
    if any(len(other.shape) != len(self.shape) for other in others):
      return None

    dims = [
        dim if all(dim == other.shape[i]
                   for other in others) else None
        for i, dim in enumerate(self.shape)
    ]
    return MockShape(*dims)

  def placeholder_value(self, placeholder_context=None):
    # [FUNCTION DOC]
    # Not supported by this mock; always raises a not-implemented error.
    raise NotImplementedError

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the text form of the dimensions tuple.
    return str(self.shape)

  def __repr__(self):
    # [FUNCTION DOC]
    # Same as the string form.
    return str(self)

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # Hash of the dimensions tuple.
    return hash(self.shape)

  def __eq__(self, other: "MockShape") -> bool:
    # [FUNCTION DOC]
    # Equal when the dimension tuples are equal (assumes the other object has a shape).
    return self.shape == other.shape


def make_shape_function_type(*shape):
  # [FUNCTION DOC]
  # HELPER: builds fixtures for the dispatch table tests.
  # Returns a function type with one required positional-only parameter named x whose type
  # constraint is a mock shape with the given dimensions.
  return function_type.FunctionType([
      function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                              False, MockShape(*shape))
  ])


class TypeDispatchTableTest(test.TestCase):
  # [CLASS DOC]
  # Tests for TypeDispatchTable behavior; each test builds its own table, no shared fixtures.

  def testVertical(self):
    # [FUNCTION DOC]
    # Adds four progressively more specific shape types and asserts the table lists them in
    # insertion order.
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None, None))
    table.add_target(make_shape_function_type(None, None, 1))
    table.add_target(make_shape_function_type(None, 1, 1))
    table.add_target(make_shape_function_type(1, 1, 1))
    self.assertEqual(
        list(table.targets), [
            make_shape_function_type(None, None, None),
            make_shape_function_type(None, None, 1),
            make_shape_function_type(None, 1, 1),
            make_shape_function_type(1, 1, 1)
        ])

  def testHorizontal(self):
    # [FUNCTION DOC]
    # Adds shapes of increasing rank and asserts all are kept in insertion order.
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(1,))
    table.add_target(make_shape_function_type(1, 2))
    table.add_target(make_shape_function_type(1, 2, 3))
    self.assertEqual(
        list(table.targets), [
            make_shape_function_type(1,),
            make_shape_function_type(1, 2),
            make_shape_function_type(1, 2, 3)
        ])

  def testDuplicateNodes(self):
    # [FUNCTION DOC]
    # Adds a target twice and asserts the table keeps a single copy at its first position.
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None))
    table.add_target(make_shape_function_type(1, None))
    table.add_target(make_shape_function_type(None, 2))
    table.add_target(make_shape_function_type(None, None))
    self.assertEqual(
        list(table.targets), [
            make_shape_function_type(None, None),
            make_shape_function_type(1, None),
            make_shape_function_type(None, 2)
        ])

  def testDeletion(self):
    # [FUNCTION DOC]
    # Deletes an existing target and asserts it disappears; deleting it again has no effect.
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None))
    table.add_target(make_shape_function_type(None, 1))
    table.add_target(make_shape_function_type(None, 2))

    self.assertEqual(
        list(table.targets), [
            make_shape_function_type(None, None),
            make_shape_function_type(None, 1),
            make_shape_function_type(None, 2)
        ])

    table.delete(make_shape_function_type(None, 2))  # Should remove the target

    self.assertEqual(
        list(table.targets), [
            make_shape_function_type(None, None),
            make_shape_function_type(None, 1),
        ])

    table.delete(make_shape_function_type(None, 2))  # Should have no effect

    self.assertEqual(
        list(table.targets), [
            make_shape_function_type(None, None),
            make_shape_function_type(None, 1),
        ])

  def testContains(self):
    # [FUNCTION DOC]
    # Asserts membership in the targets view is true for added types and false for types that
    # were never added, including similar-looking ones.
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None, None))
    table.add_target(make_shape_function_type(None, 1))
    table.add_target(make_shape_function_type(1, 1))
    table.add_target(make_shape_function_type(None, 2, 1))

    self.assertIn(make_shape_function_type(None, None, None), table.targets)
    self.assertIn(make_shape_function_type(None, 1), table.targets)
    self.assertIn(make_shape_function_type(1, 1), table.targets)
    self.assertIn(make_shape_function_type(None, 2, 1), table.targets)

    self.assertNotIn(make_shape_function_type(None, None, 1), table.targets)
    self.assertNotIn(make_shape_function_type(1, None), table.targets)
    self.assertNotIn(make_shape_function_type(1, 2), table.targets)
    self.assertNotIn(make_shape_function_type(None, 2, None), table.targets)

  def testDispatchExactMatches(self):
    # [FUNCTION DOC]
    # Asserts dispatching a type that is itself a target returns that same target.
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None, None))
    table.add_target(make_shape_function_type(None, 1, None))
    table.add_target(make_shape_function_type(None, 1, 2))
    table.add_target(make_shape_function_type(None, 2, 2))

    self.assertEqual(
        table.dispatch(make_shape_function_type(None, 1, 2)),
        make_shape_function_type(None, 1, 2))
    self.assertEqual(
        table.dispatch(make_shape_function_type(None, 1, None)),
        make_shape_function_type(None, 1, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(None, None, None)),
        make_shape_function_type(None, None, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(None, 2, 2)),
        make_shape_function_type(None, 2, 2))

  def testDispatchMoreSpecific(self):
    # [FUNCTION DOC]
    # Dispatches fully known shapes and asserts the closest (most specific) compatible target
    # is chosen among several wider ones.
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None, None))
    table.add_target(make_shape_function_type(None, 1, None))
    table.add_target(make_shape_function_type(None, 1, 2))
    table.add_target(make_shape_function_type(None, 2, 2))

    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, 1, 2))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 3)),
        make_shape_function_type(None, 1, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 3, 3)),
        make_shape_function_type(None, None, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 2, 2)),
        make_shape_function_type(None, 2, 2))

  def testDispatchNoMatches(self):
    # [FUNCTION DOC]
    # Dispatches shapes of mismatched rank and asserts nothing is returned.
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, 1, None))
    table.add_target(make_shape_function_type(None, 1, 2))
    table.add_target(make_shape_function_type(None, 2, 2))

    self.assertIsNone(table.dispatch(make_shape_function_type(1, 2)))
    self.assertIsNone(table.dispatch(make_shape_function_type(1, 2, 3)))
    self.assertIsNone(table.dispatch(make_shape_function_type(1, 2, 3, 4)))

  def testDispatchCachedAddUpdates(self):
    # [FUNCTION DOC]
    # Repeatedly dispatches the same request while adding progressively more specific targets;
    # asserts the cached result is updated to the newly added closer target each time.
    table = type_dispatch.TypeDispatchTable()

    table.add_target(make_shape_function_type(None, None, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, None, None))

    table.add_target(make_shape_function_type(None, 1, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, 1, None))

    table.add_target(make_shape_function_type(None, 1, 2))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, 1, 2))

    table.add_target(make_shape_function_type(1, 1, 2))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(1, 1, 2))

  def testDispatchCachedDeleteUpdates(self):
    # [FUNCTION DOC]
    # Dispatches a request, then deletes the chosen target repeatedly; asserts each dispatch
    # falls back to the next closest remaining target (cache entries are invalidated).
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None, None))
    table.add_target(make_shape_function_type(None, 1, None))
    table.add_target(make_shape_function_type(None, 1, 2))
    table.add_target(make_shape_function_type(1, 1, 2))

    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(1, 1, 2))

    table.delete(make_shape_function_type(1, 1, 2))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, 1, 2))

    table.delete(make_shape_function_type(None, 1, 2))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, 1, None))

    table.delete(make_shape_function_type(None, 1, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, None, None))

  def testDispatchCacheOrderingDeterminism(self):
    # [FUNCTION DOC]
    # Builds three tables with the same targets inserted in different orders; asserts the
    # target sets are equal but a request matching several unrelated targets dispatches to the
    # earliest inserted one in each table.
    table_1 = type_dispatch.TypeDispatchTable()
    table_1.add_target(make_shape_function_type(1, None, None))
    table_1.add_target(make_shape_function_type(None, 2, None))
    table_1.add_target(make_shape_function_type(None, None, 3))

    table_2 = type_dispatch.TypeDispatchTable()
    table_2.add_target(make_shape_function_type(None, 2, None))
    table_2.add_target(make_shape_function_type(1, None, None))
    table_2.add_target(make_shape_function_type(None, None, 3))

    table_3 = type_dispatch.TypeDispatchTable()
    table_3.add_target(make_shape_function_type(None, None, 3))
    table_3.add_target(make_shape_function_type(1, None, None))
    table_3.add_target(make_shape_function_type(None, 2, None))

    # table_1, table_2, table_3 have the same targets
    self.assertEqual(set(table_1.targets), set(table_2.targets))
    self.assertEqual(set(table_2.targets), set(table_3.targets))

    # But they dispatch to the first target they find which does not have any
    # more specific viable target.
    shape = make_shape_function_type(1, 2, 3)
    self.assertEqual(
        table_1.dispatch(shape), make_shape_function_type(1, None, None))
    self.assertEqual(
        table_2.dispatch(shape), make_shape_function_type(None, 2, None))
    self.assertEqual(
        table_3.dispatch(shape), make_shape_function_type(None, None, 3))

  def testGeneralizedExisting(self):
    # [FUNCTION DOC]
    # Asserts generalization of a type whose wider form is already a target returns that wide
    # fully unknown type.
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None, None))
    table.add_target(make_shape_function_type(None, 1, None))
    table.add_target(make_shape_function_type(None, 1, 2))
    self.assertEqual(
        table.try_generalizing_function_type(
            make_shape_function_type(None, 1, 3)),
        make_shape_function_type(None, None, None))

  def testGeneralizedNovel(self):
    # [FUNCTION DOC]
    # Asserts generalization can yield a new type not in the table by merging common
    # dimensions across targets, giving the fully unknown shape.
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, 1, None))
    table.add_target(make_shape_function_type(None, 1, 2))
    self.assertEqual(
        table.try_generalizing_function_type(
            make_shape_function_type(None, 2, 3)),
        make_shape_function_type(None, None, None))

  def testGeneralizedUnknown(self):
    # [FUNCTION DOC]
    # Asserts that when no target has a compatible rank, generalization returns the given type
    # unchanged.
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, 1))
    table.add_target(make_shape_function_type(None, 2))
    table.add_target(make_shape_function_type(None, 3))
    self.assertEqual(
        table.try_generalizing_function_type(
            make_shape_function_type(None, 4, 3)),
        make_shape_function_type(None, 4, 3))


if __name__ == "__main__":
  test.main()
